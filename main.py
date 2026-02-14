"""Main entry point for the Polymarket Quant Trading System."""
import asyncio
import logging
import sys
from pathlib import Path

from src.core.orchestrator import TradingOrchestrator
from src.utils.logger import setup_logger
import config


async def main():
    """Initialize and run the trading system."""
    # Setup directories
    Path(config.MODEL_PATH).mkdir(parents=True, exist_ok=True)
    Path(config.LOG_PATH).mkdir(parents=True, exist_ok=True)
    Path('data').mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logger('main', config.LOG_PATH + 'main.log')
    logger.info("="*80)
    logger.info("Polymarket Quant Trading System - Starting...")
    logger.info("="*80)
    
    # Validate configuration
    if not config.COINBASE_API_KEY or not config.DISCORD_WEBHOOK_URL:
        logger.error("Missing required API keys. Check your .env file.")
        sys.exit(1)
    
    # Initialize the orchestrator
    orchestrator = TradingOrchestrator()
    
    try:
        # Start the trading system
        await orchestrator.start()
    except KeyboardInterrupt:
        logger.info("Shutdown signal received...")
    except Exception as e:
        logger.exception(f"Critical error in main loop: {e}")
    finally:
        await orchestrator.stop()
        logger.info("System shutdown complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested... exiting gracefully.")
        sys.exit(0)
