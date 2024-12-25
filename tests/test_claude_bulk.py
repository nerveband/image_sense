"""
Test bulk image processing with Claude
"""

import os
import pytest
import logging
from pathlib import Path
from dotenv import load_dotenv
from src.core.llm_handler import LLMHandler, LLMProvider

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

@pytest.fixture
def claude_api_key() -> str:
    """Get Claude API key from environment"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not found in environment variables")
    return api_key

@pytest.fixture
def test_images() -> list[str]:
    """Get list of test image paths"""
    test_dir = Path("tests/test_images/bulk_images_5")
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend([str(p) for p in test_dir.glob(ext)])
    # Exclude .DS_Store
    image_paths = [p for p in image_paths if not p.endswith('.DS_Store')]
    return sorted(image_paths)[:3]  # Limit to 3 images for faster testing

@pytest.mark.asyncio
async def test_claude_bulk_processing(claude_api_key: str, test_images: list[str]):
    """Test bulk processing with Claude"""
    
    # Initialize handler
    handler = LLMHandler(
        provider=LLMProvider.CLAUDE,
        model="claude-3-sonnet-20240229",  # Use Sonnet model which supports vision
        api_key=claude_api_key,
        compress_images=True  # Enable compression since some images are large
    )
    
    # Process images
    results = await handler.process_batch(test_images)
    
    # Print results
    print("\nProcessing Results:")
    print("-" * 50)
    
    success_count = 0
    for result in results:
        path = Path(result['path']).name
        if result['success']:
            success_count += 1
            print(f"\n✅ {path}:")
            metadata = result['metadata']
            print(f"Description: {metadata['description'][:100]}...")
            print(f"Keywords: {', '.join(metadata['keywords'])}")
            print(f"Categories: {', '.join(metadata['categories'])}")
        else:
            print(f"\n❌ {path}:")
            print(f"Error: {result['error']}")
    
    print("\nSummary:")
    print(f"Total images: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(results) - success_count}")
    
    # Get processing stats
    stats = handler.get_processing_stats()
    if stats:
        print("\nProcessing Stats:")
        print(f"Average time per image: {stats.get('avg_time_per_image', 0):.2f}s")
        if stats.get('compression_savings'):
            print(f"Time saved with compression: {stats['compression_savings']:.2f}s per image")
    
    # Verify results
    assert len(results) == len(test_images), "Not all images were processed"
    assert success_count > 0, "No images were processed successfully"

if __name__ == "__main__":
    pytest.main(["-v", __file__]) 