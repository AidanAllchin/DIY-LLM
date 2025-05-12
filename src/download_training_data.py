#!/usr/bin/env python3
import os
import json
import asyncio
import argparse
import random
import gzip
import glob
from typing import List, Tuple, Optional

import aiohttp
from rich.progress import (
    Progress, TaskID, TextColumn, 
    BarColumn, TransferSpeedColumn, 
    TimeRemainingColumn, FileSizeColumn,
    SpinnerColumn, ProgressColumn
)
from rich.text import Text
from rich.live import Live
from rich.console import Group

class FileCountColumn(ProgressColumn):
    """Renders file count for overall progress."""
    
    def render(self, task) -> Text:
        """Show completed/total files."""
        completed = task.completed
        total = task.total
        return Text(f"{completed}/{total} files", style="bold green")

# Create progress instances for different types of tasks
def create_progress(description_style, include_file_count=False):
    """Create a configurable progress instance"""
    columns = [
        SpinnerColumn(),
        TextColumn(f"[{description_style}]{{task.description}}"),
        BarColumn(),
        FileSizeColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn()
    ]
    
    if include_file_count:
        columns = [
            SpinnerColumn(),
            TextColumn(f"[{description_style}]{{task.description}}"),
            BarColumn(),
            FileCountColumn(),
            TimeRemainingColumn()
        ]
    
    return Progress(*columns)

def extract_json_objects(text: str):
    """Extract JSON objects from text that might contain multiple concatenated objects."""
    objects = []
    decoder = json.JSONDecoder()
    idx = 0
    
    while idx < len(text):
        text_slice = text[idx:].lstrip()
        if not text_slice:
            break
            
        try:
            obj, end_idx = decoder.raw_decode(text_slice)
            objects.append(obj)
            idx += (text_slice.find(text_slice.lstrip()[0]) + end_idx)
        except json.JSONDecodeError:
            # Skip to the next potential JSON object
            idx += 1
    
    return objects

def split_file_by_json_objects(
        filepath: str, 
        max_size_mb: int = 512, 
        output_dir: Optional[str] = None
    ) -> List[str]:
    """
    Split a large JSON file into smaller files containing whole JSON objects
    
    Args:
        filepath: Path to the JSON file to split
        max_size_mb: Maximum size of each split file in MB
        output_dir: Directory where to save split files (default: same as input file)
        
    Returns:
        List of paths to the split files
    """
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(filepath)
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename for creating split files
    base_filename = os.path.basename(filepath)
    name, ext = os.path.splitext(base_filename)
    
    # Convert max size to bytes
    max_size_bytes = max_size_mb * 1024 * 1024
    
    print(f"Splitting {filepath} into chunks of max {max_size_mb}MB...")
    
    # Read file in chunks to avoid loading everything into memory
    json_objects = []
    current_file_size = 0
    part_num = 1
    output_files = []
    
    try:
        # Read and parse in reasonable chunks
        buffer_size = 10 * 1024 * 1024  # 10MB buffer
        
        with open(filepath, 'r', encoding='utf-8') as file:
            # First step: read buffer-sized chunks and extract JSON objects
            buffer = ""
            objects_collection = []
            
            while True:
                chunk = file.read(buffer_size)
                if not chunk:
                    break
                    
                buffer += chunk
                
                # Extract complete JSON objects from buffer
                try:
                    while True:
                        try:
                            decoder = json.JSONDecoder()
                            obj, idx = decoder.raw_decode(buffer.lstrip())
                            # Found a complete object
                            objects_collection.append(obj)
                            obj_str = json.dumps(obj)
                            current_file_size += len(obj_str.encode('utf-8')) + 1
                            # Remove the processed part from buffer
                            buffer = buffer.lstrip()[idx:].lstrip()
                        except json.JSONDecodeError:
                            break
                except Exception as e:
                    print(f"Error parsing JSON in buffer: {e}")

                # When accumulated size exceeds max_size_bytes, write to file
                if current_file_size >= max_size_bytes:
                    output_path = os.path.join(output_dir, f"{name}_part{part_num:03d}{ext}")
                    
                    with open(output_path, 'w', encoding='utf-8') as out_file:
                        # Write JSON array
                        out_file.write('[\n')
                        for i, obj in enumerate(objects_collection):
                            json_str = json.dumps(obj)
                            out_file.write(json_str)
                            if i < len(objects_collection) - 1:
                                out_file.write(',\n')
                        out_file.write('\n]')
                    
                    output_files.append(output_path)
                    objects_collection = []
                    current_file_size = 0
                    part_num += 1
            
            # Process any remaining objects in the buffer
            if buffer.strip():
                try:
                    remaining_objects = extract_json_objects(buffer)
                    objects_collection.extend(remaining_objects)
                except Exception as e:
                    print(f"Error extracting remaining JSON objects: {e}")
            
            # Write any remaining objects to file
            if objects_collection:
                output_path = os.path.join(output_dir, f"{name}_part{part_num:03d}{ext}")
                
                with open(output_path, 'w', encoding='utf-8') as out_file:
                    # Write JSON array
                    out_file.write('[\n')
                    for i, obj in enumerate(objects_collection):
                        json_str = json.dumps(obj)
                        out_file.write(json_str)
                        if i < len(objects_collection) - 1:
                            out_file.write(',\n')
                    out_file.write('\n]')
                
                output_files.append(output_path)
            
        # Only remove the original file if split was successful and produced output files
        if output_files:
            os.remove(filepath)
            print(f"Split {filepath} into {len(output_files)} files")
        else:
            print(f"Warning: Split operation did not produce any output files for {filepath}")
            
        return output_files
        
    except Exception as e:
        print(f"Error splitting file {filepath}: {e}")
        return []

async def download_file(
        session: aiohttp.ClientSession, 
        url: str, 
        target_dir: str, 
        semaphore: asyncio.Semaphore, 
        progress: Progress, 
        task_id: TaskID
    ) -> Tuple[bool, Optional[str]]:
    """Download a single file asynchronously"""
    async with semaphore:
        try:
            filename = os.path.basename(url)
            filepath = os.path.join(target_dir, filename)
            
            async with session.get(url.strip()) as response:
                if response.status != 200:
                    print(f"Failed to download {url}: HTTP {response.status}")
                    return False, None
                
                total_size = int(response.headers.get('content-length', 0))
                if total_size == 0:
                    total_size = 1  # Avoid division by zero
                
                progress.update(task_id, total=total_size)
                
                with open(filepath, 'wb') as f:
                    downloaded = 0
                    while True:
                        chunk = await response.content.read(1024 * 1024)  # 1MB chunks
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress.update(task_id, completed=downloaded)
            
            # Get the actual file size after download
            actual_size = os.path.getsize(filepath)
            progress.update(task_id, completed=actual_size, total=actual_size)
            
            return True, filepath
        except Exception as e:
            print(f"Error downloading {url}: {str(e)}")
            return False, None

async def extract_gz_file(
        filepath: str, 
        progress: Progress, 
        task_id: TaskID,
        max_file_size_mb: int = 512
    ) -> List[str]:
    """Extract a .gz file"""
    try:
        if not filepath.endswith('.gz'):
            return [filepath] if os.path.exists(filepath) else []
            
        output_path = filepath[:-3]  # Remove .gz extension
        
        # Get file size for progress tracking
        file_size = os.path.getsize(filepath)
        progress.update(task_id, total=file_size)
        
        with gzip.open(filepath, 'rb') as f_in:
            processed = 0
            with open(output_path, 'wb') as f_out:
                while True:
                    chunk = f_in.read(1024 * 1024)  # 1MB chunks
                    if not chunk:
                        break
                    f_out.write(chunk)
                    processed += len(chunk)
                    progress.update(task_id, completed=min(processed, file_size))
        
        # Delete the compressed file
        os.remove(filepath)
        print(f"Extracted {filepath} to {output_path}")
            
        # Split the file if it's too large and is a JSON file
        if output_path.endswith('.json') and os.path.getsize(output_path) > max_file_size_mb * 1024 * 1024:
            print(f"File {output_path} is large ({os.path.getsize(output_path) / (1024 * 1024):.2f}MB). Splitting...")
            split_files = split_file_by_json_objects(output_path, max_file_size_mb)
            return split_files
            
        return [output_path]
    except Exception as e:
        print(f"Error extracting {filepath}: {str(e)}")
        return []

async def download_and_extract(
        urls: List[str], 
        target_dir: str, 
        max_parallel: int,
        extract: bool = True,
        max_file_size_mb: int = 512
    ):
    """Download all files in parallel and extract them if requested"""
    os.makedirs(target_dir, exist_ok=True)
    
    # Create a semaphore to limit concurrent downloads
    semaphore = asyncio.Semaphore(max_parallel)
    
    # Set up progress displays
    download_progress = create_progress("bold blue")
    overall_progress = create_progress("bold green", include_file_count=True)
    progress_group = Group(download_progress, overall_progress)
    
    downloaded_files = []
    
    # Skip already processed files
    unprocessed_urls = []
    for url in urls:
        filename = os.path.basename(url)
        extracted = filename[:-3] if filename.endswith('.gz') else filename
        name, ext = os.path.splitext(extracted)
        pattern = os.path.join(target_dir, f"{name}_part*{ext}")
        if extract and glob.glob(pattern):
            print(f"Skipping {filename}, part files already exist.")
        elif not extract and os.path.exists(os.path.join(target_dir, filename)):
            print(f"Skipping {filename}, file already exists.")
        else:
            unprocessed_urls.append(url)
    if not unprocessed_urls:
        print("No new files to download.")
        return

    urls = unprocessed_urls

    # Download files
    with Live(progress_group, refresh_per_second=10):
        # Create overall progress task
        overall_task_id = overall_progress.add_task(
            f"Overall progress (0/{len(urls)} files)", 
            total=len(urls),
            completed=0
        )
        
        async with aiohttp.ClientSession() as session:
            # Create download tasks
            download_tasks = []
            
            for url in urls:
                filename = os.path.basename(url)
                task_id = download_progress.add_task(f"Downloading {filename}", total=0)
                download_tasks.append(
                    download_file(
                        session=session, 
                        url=url, 
                        target_dir=target_dir, 
                        semaphore=semaphore, 
                        progress=download_progress, 
                        task_id=task_id
                    )
                )
            
            # Process downloads as they complete
            completed = 0
            
            for task_future in asyncio.as_completed(download_tasks):
                success, filepath = await task_future
                completed += 1
                
                # Update overall progress
                overall_progress.update(
                    overall_task_id, 
                    completed=completed, 
                    description=f"Overall progress ({completed}/{len(urls)} files)"
                )
                
                # Track successful downloads
                if success and filepath:
                    downloaded_files.append(filepath)
    
    # Extract files if requested
    if extract and downloaded_files:
        print("\nStarting extraction phase...")
        
        # Filter for .gz files
        gz_files = [f for f in downloaded_files if f.endswith('.gz')]
        
        if not gz_files:
            print("No .gz files to extract.")
            return
        
        # Create progress displays for extraction
        extract_progress = create_progress("bold magenta")
        extract_overall = create_progress("bold yellow", include_file_count=True)
        extract_group = Group(extract_progress, extract_overall)
        
        with Live(extract_group, refresh_per_second=10):
            # Add overall extraction progress task
            overall_extract_id = extract_overall.add_task(
                f"Overall extraction (0/{len(gz_files)} files)", 
                total=len(gz_files),
                completed=0
            )
            
            # Create extraction tasks
            extract_tasks = []
            
            for filepath in gz_files:
                filename = os.path.basename(filepath)
                task_id = extract_progress.add_task(f"Extracting {filename}", total=0)
                extract_tasks.append(
                    extract_gz_file(
                        filepath=filepath,
                        progress=extract_progress,
                        task_id=task_id,
                        max_file_size_mb=max_file_size_mb
                    )
                )
            
            # Wait for all extraction tasks to complete
            results = await asyncio.gather(*extract_tasks)
            
            # Update progress to show completion
            extract_overall.update(
                overall_extract_id,
                completed=len(gz_files),
                description=f"Overall extraction ({len(gz_files)}/{len(gz_files)} files)"
            )
            
            # Flatten the list of file paths (since extract_gz_file now returns a list)
            extracted_files = [file for result in results for file in result]
            
            # Report extraction results
            successful = len(extracted_files)
            print(f"Extracted {len(gz_files)} files into {successful} files successfully")

async def fetch_url_list(version: str) -> List[str]:
    """Fetch the URL list from HuggingFace"""
    url = f"https://huggingface.co/datasets/allenai/dolma/raw/main/urls/{version}.txt"
    
    print(f"Fetching URL list from {url}")
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise Exception(f"Failed to fetch URL list: HTTP {response.status}")
            
            content = await response.text()
            return [line.strip() for line in content.splitlines() if line.strip()]

def main():
    parser = argparse.ArgumentParser(description="Download Dolma dataset using async Python")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory to store downloaded data")
    parser.add_argument("--parallel", type=int, default=4, help="Number of parallel downloads")
    parser.add_argument("--version", type=str, default="v1_7", help="Version of Dolma to download")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of files to download")
    parser.add_argument("--random", action="store_true", help="Randomly select files when using --limit")
    parser.add_argument("--extract", action="store_true", help="Extract .gz files after downloading")
    parser.add_argument("--max-file-size-mb", type=int, default=512, help="Maximum size of files to extract in MB (files larger than this will be split into smaller files)")
    
    args = parser.parse_args()
    
    try:
        # Fetch the URL list
        urls = asyncio.run(fetch_url_list(args.version))
        
        total_urls = len(urls)
        print(f"Found {total_urls} files in the URL list")
        
        # Apply limit if specified
        if args.limit is not None and args.limit < total_urls:
            if args.random:
                print(f"Randomly selecting {args.limit} files out of {total_urls}")
                urls = random.sample(urls, args.limit)
            else:
                print(f"Taking the first {args.limit} files out of {total_urls}")
                urls = urls[:args.limit]
        
        # Download and extract files
        asyncio.run(download_and_extract(
            urls=urls, 
            target_dir=args.data_dir, 
            max_parallel=args.parallel,
            extract=args.extract,
            max_file_size_mb=args.max_file_size_mb
        ))
        
        print("Process completed!")
            
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    #random.seed(42)  # For reproducibility
    exit(main())
