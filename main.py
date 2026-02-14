"""Main entry point for Polymarket Quantitative Trading System."""

import asyncio
import signal
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

import config
from src.utils.logger import setup_logger
from src.utils.database import DatabaseManager
from src.data_ingestion.coinbase_stream import CoinbaseStream
from src.data_ingestion.polymarket_stream import PolymarketStream
from src.data_ingestion.data_synchronizer import DataSynchronizer
from src.features.feature_pipeline import FeaturePipeline
from src.ml_engine.ensemble_model import EnsembleModel
from src.ml_engine.model_trainer import ModelTrainer
from src.execution.paper_trading import PaperTradingEngine
from src.execution.discord_notifier import DiscordNotifier

console = Console()


class TradingSystem:
    """Main trading system orchestrator."""

    def __init__(self):
        """Initialize the trading system."""
        self.running = False
        self.tasks = []
        
        # Components
        self.db_manager = None
        self.coinbase_stream = None
        self.polymarket_stream = None
        self.data_synchronizer = None
        self.feature_pipeline = None
        self.ensemble_model = None
        self.model_trainer = None
        self.paper_trading = None
        self.discord_notifier = None

    async def initialize(self):
        """Initialize all system components."""
        logger.info("Initializing Polymarket Quantitative Trading System...")

        try:
            # Validate configuration
            config.validate_config()
            logger.info("✅ Configuration validated")

            # Initialize database
            self.db_manager = DatabaseManager()
            await self.db_manager.initialize()
            logger.info("✅ Database initialized")

            # Initialize Discord notifier
            self.discord_notifier = DiscordNotifier()
            await self.discord_notifier.send_startup_message()
            logger.info("✅ Discord notifier initialized")

            # Initialize data streams
            self.coinbase_stream = CoinbaseStream()
            self.polymarket_stream = PolymarketStream()
            logger.info("✅ Data streams initialized")

            # Initialize data synchronizer
            self.data_synchronizer = DataSynchronizer(
                self.coinbase_stream, self.polymarket_stream
            )
            logger.info("✅ Data synchronizer initialized")

            # Initialize feature pipeline
            self.feature_pipeline = FeaturePipeline()
            logger.info("✅ Feature pipeline initialized")

            # Initialize ML models
            self.ensemble_model = EnsembleModel()
            self.model_trainer = ModelTrainer(
                self.ensemble_model, self.db_manager, self.discord_notifier
            )
            logger.info("✅ ML engine initialized")

            # Initialize paper trading engine
            self.paper_trading = PaperTradingEngine(
                self.db_manager, self.discord_notifier
            )
            logger.info("✅ Paper trading engine initialized")

            logger.info("🚀 System initialization complete!")
            return True

        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            if self.discord_notifier:
                await self.discord_notifier.send_error(f"System initialization failed: {e}")
            return False

    async def start(self):
        """Start the trading system."""
        self.running = True
        logger.info("Starting trading system...")

        try:
            # Start data streams
            self.tasks.append(asyncio.create_task(self.coinbase_stream.start()))
            self.tasks.append(asyncio.create_task(self.polymarket_stream.start()))
            logger.info("✅ Data streams started")

            # Start data synchronizer
            self.tasks.append(asyncio.create_task(self.data_synchronizer.start()))
            logger.info("✅ Data synchronizer started")

            # Start model trainer (auto-retrain loop)
            self.tasks.append(asyncio.create_task(self.model_trainer.start_training_loop()))
            logger.info("✅ Model training loop started")

            # Start main trading loop
            self.tasks.append(asyncio.create_task(self.trading_loop()))
            logger.info("✅ Trading loop started")

            # Wait for all tasks
            await asyncio.gather(*self.tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"❌ Error in trading system: {e}")
            await self.stop()

    async def trading_loop(self):
        """Main trading loop."""
        logger.info("Trading loop started")

        while self.running:
            try:
                # Get synchronized data
                sync_data = await self.data_synchronizer.get_latest_data()

                if sync_data is None:
                    await asyncio.sleep(1)
                    continue

                # Generate features
                features = await self.feature_pipeline.generate_features(sync_data)

                if features is None or len(features) < config.MIN_DATA_POINTS:
                    await asyncio.sleep(1)
                    continue

                # Get prediction from ensemble model
                prediction = await self.ensemble_model.predict(features)

                if prediction is None:
                    await asyncio.sleep(1)
                    continue

                # Check if prediction meets confidence threshold
                if prediction["confidence"] >= config.MIN_CONFIDENCE:
                    # Execute paper trade
                    await self.paper_trading.execute_signal(
                        prediction, sync_data, features
                    )

                await asyncio.sleep(0.1)  # Small delay to prevent CPU overload

            except asyncio.CancelledError:
                logger.info("Trading loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying

    async def stop(self):
        """Stop the trading system gracefully."""
        logger.info("Stopping trading system...")
        self.running = False

        # Cancel all tasks
        for task in self.tasks:
            task.cancel()

        # Stop components
        if self.coinbase_stream:
            await self.coinbase_stream.stop()
        if self.polymarket_stream:
            await self.polymarket_stream.stop()
        if self.data_synchronizer:
            await self.data_synchronizer.stop()
        if self.model_trainer:
            await self.model_trainer.stop()

        # Send shutdown notification
        if self.discord_notifier:
            await self.discord_notifier.send_shutdown_message()

        # Close database
        if self.db_manager:
            await self.db_manager.close()

        logger.info("✅ Trading system stopped")


def print_banner():
    """Print startup banner."""
    banner = Text()
    banner.append("\n")
    banner.append("═" * 70, style="cyan")
    banner.append("\n")
    banner.append("  POLYMARKET QUANTITATIVE TRADING SYSTEM  ", style="bold cyan")
    banner.append("\n")
    banner.append("  Phase 1: Paper Trading & ML Engine  ", style="green")
    banner.append("\n")
    banner.append("═" * 70, style="cyan")
    banner.append("\n\n")
    
    panel = Panel(
        banner,
        border_style="cyan",
        expand=False,
    )
    console.print(panel)


async def main():
    """Main entry point."""
    # Setup logging
    setup_logger()

    # Print banner
    print_banner()

    # Create trading system
    system = TradingSystem()

    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        asyncio.create_task(system.stop())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize system
    if not await system.initialize():
        logger.error("Failed to initialize system. Exiting.")
        sys.exit(1)

    # Start system
    try:
        await system.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        await system.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Trading system stopped by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Fatal error: {e}[/red]")
        sys.exit(1)