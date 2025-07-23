"""
Coordinate caching system for efficient spatial data access.

This module provides intelligent caching of coordinate data to minimize
database queries and improve performance for spatial operations.
"""

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CoordinateInfo:
    """Information about a coordinate entry."""

    frame_id: int
    x: int
    y: int
    z: int
    is_maldi: bool = False


class CoordinateCache:
    """
    Efficient coordinate caching system with lazy loading.

    This class provides fast coordinate lookups with minimal memory usage
    by implementing intelligent caching strategies.
    """

    def __init__(self, db_path: Path, preload_all: bool = False):
        """
        Initialize the coordinate cache.

        Args:
            db_path: Path to the SQLite database file
            preload_all: Whether to preload all coordinates immediately
        """
        self.db_path = Path(db_path)
        self._coordinates: Dict[int, CoordinateInfo] = {}
        self._dimensions: Optional[Tuple[int, int, int]] = None
        self._is_maldi: Optional[bool] = None
        self._loaded_ranges: Set[Tuple[int, int]] = set()

        # Check if this is a MALDI dataset
        self._detect_maldi_format()

        if preload_all:
            self._preload_all_coordinates()

        logger.info(
            f"Initialized CoordinateCache for {'MALDI' if self._is_maldi else 'non-MALDI'} dataset"
        )

    def _detect_maldi_format(self) -> None:
        """Detect if this is a MALDI dataset by checking for MaldiFrameInfo table."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()

                # Check if MaldiFrameInfo table exists
                cursor.execute(
                    """
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name='MaldiFrameInfo'
                """
                )

                self._is_maldi = cursor.fetchone() is not None

        except Exception as e:
            logger.warning(f"Error detecting MALDI format: {e}")
            self._is_maldi = False

    def _preload_all_coordinates(self) -> None:
        """Preload all coordinates for maximum performance."""
        logger.info("Preloading all coordinates")

        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()

                if self._is_maldi:
                    # Load MALDI coordinates
                    cursor.execute(
                        """
                        SELECT Frame, XIndexPos, YIndexPos
                        FROM MaldiFrameInfo
                        ORDER BY Frame
                    """
                    )

                    for frame_id, x, y in cursor.fetchall():
                        self._coordinates[frame_id] = CoordinateInfo(
                            frame_id=frame_id, x=int(x), y=int(y), z=0, is_maldi=True
                        )
                else:
                    # For non-MALDI data, generate linear coordinates
                    cursor.execute("SELECT COUNT(*) FROM Frames")
                    frame_count = cursor.fetchone()[0]

                    for frame_id in range(1, frame_count + 1):
                        self._coordinates[frame_id] = CoordinateInfo(
                            frame_id=frame_id,
                            x=frame_id - 1,  # 0-based
                            y=0,
                            z=0,
                            is_maldi=False,
                        )

                logger.info(f"Preloaded {len(self._coordinates)} coordinates")

        except Exception as e:
            logger.error(f"Error preloading coordinates: {e}")
            self._coordinates.clear()

    def _load_coordinate_range(self, start_frame: int, end_frame: int) -> None:
        """
        Load coordinates for a specific frame range.

        Args:
            start_frame: Starting frame ID (inclusive)
            end_frame: Ending frame ID (inclusive)
        """
        # Check if this range is already loaded
        range_key = (start_frame, end_frame)
        if range_key in self._loaded_ranges:
            return

        logger.debug(f"Loading coordinates for frames {start_frame}-{end_frame}")

        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()

                if self._is_maldi:
                    cursor.execute(
                        """
                        SELECT Frame, XIndexPos, YIndexPos
                        FROM MaldiFrameInfo
                        WHERE Frame BETWEEN ? AND ?
                        ORDER BY Frame
                    """,
                        (start_frame, end_frame),
                    )

                    for frame_id, x, y in cursor.fetchall():
                        if frame_id not in self._coordinates:
                            self._coordinates[frame_id] = CoordinateInfo(
                                frame_id=frame_id,
                                x=int(x),
                                y=int(y),
                                z=0,
                                is_maldi=True,
                            )
                else:
                    # For non-MALDI data, generate coordinates
                    for frame_id in range(start_frame, end_frame + 1):
                        if frame_id not in self._coordinates:
                            self._coordinates[frame_id] = CoordinateInfo(
                                frame_id=frame_id,
                                x=frame_id - 1,  # 0-based
                                y=0,
                                z=0,
                                is_maldi=False,
                            )

                self._loaded_ranges.add(range_key)
                logger.debug(f"Loaded {end_frame - start_frame + 1} coordinates")

        except Exception as e:
            logger.error(f"Error loading coordinate range: {e}")

    def get_coordinate(self, frame_id: int) -> Optional[Tuple[int, int, int]]:
        """
        Get coordinates for a specific frame ID.

        Args:
            frame_id: Frame ID to look up

        Returns:
            Tuple of (x, y, z) coordinates using 0-based indexing, or None if not found
        """
        # Check if coordinate is already cached
        if frame_id in self._coordinates:
            coord_info = self._coordinates[frame_id]
            return (coord_info.x, coord_info.y, coord_info.z)

        # Try to load a small range around this frame
        batch_size = 100
        start_frame = max(1, frame_id - batch_size // 2)
        end_frame = frame_id + batch_size // 2

        self._load_coordinate_range(start_frame, end_frame)

        # Check again after loading
        if frame_id in self._coordinates:
            coord_info = self._coordinates[frame_id]
            return (coord_info.x, coord_info.y, coord_info.z)

        logger.warning(f"Coordinate not found for frame {frame_id}")
        return None

    def get_coordinates_batch(
        self, frame_ids: List[int]
    ) -> Dict[int, Tuple[int, int, int]]:
        """
        Get coordinates for multiple frame IDs efficiently.

        Args:
            frame_ids: List of frame IDs to look up

        Returns:
            Dictionary mapping frame IDs to (x, y, z) coordinates
        """
        result = {}
        missing_frames = []

        # Check what's already cached
        for frame_id in frame_ids:
            if frame_id in self._coordinates:
                coord_info = self._coordinates[frame_id]
                result[frame_id] = (coord_info.x, coord_info.y, coord_info.z)
            else:
                missing_frames.append(frame_id)

        # Load missing frames in ranges
        if missing_frames:
            missing_frames.sort()

            # Group consecutive frames into ranges
            ranges = []
            start = missing_frames[0]
            end = start

            for frame_id in missing_frames[1:]:
                if frame_id == end + 1:
                    end = frame_id
                else:
                    ranges.append((start, end))
                    start = frame_id
                    end = frame_id
            ranges.append((start, end))

            # Load each range
            for start_frame, end_frame in ranges:
                self._load_coordinate_range(start_frame, end_frame)

            # Add newly loaded coordinates to result
            for frame_id in missing_frames:
                if frame_id in self._coordinates:
                    coord_info = self._coordinates[frame_id]
                    result[frame_id] = (coord_info.x, coord_info.y, coord_info.z)

        return result

    def get_dimensions(self) -> Tuple[int, int, int]:
        """
        Calculate dataset dimensions from coordinate data.

        Returns:
            Tuple of (x_size, y_size, z_size)
        """
        if self._dimensions is not None:
            return self._dimensions

        # Ensure we have at least some coordinates loaded
        if not self._coordinates:
            try:
                with sqlite3.connect(str(self.db_path)) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT MIN(Id), MAX(Id) FROM Frames")
                    min_frame, max_frame = cursor.fetchone()

                    if min_frame and max_frame:
                        self._load_coordinate_range(min_frame, max_frame)
            except Exception as e:
                logger.error(f"Error loading frames for dimension calculation: {e}")

        if not self._coordinates:
            logger.warning("No coordinates available for dimension calculation")
            return (1, 1, 1)

        # Calculate dimensions from all coordinates
        x_coords = [coord.x for coord in self._coordinates.values()]
        y_coords = [coord.y for coord in self._coordinates.values()]
        z_coords = [coord.z for coord in self._coordinates.values()]

        if not x_coords:
            return (1, 1, 1)

        # Calculate actual dimensions using min/max bounds (not assuming 0-based)
        x_size = max(x_coords) - min(x_coords) + 1
        y_size = max(y_coords) - min(y_coords) + 1
        z_size = max(z_coords) - min(z_coords) + 1

        self._dimensions = (x_size, y_size, z_size)
        logger.info(f"Calculated dimensions: {self._dimensions}")

        return self._dimensions

    def is_maldi_dataset(self) -> bool:
        """Check if this is a MALDI dataset."""
        return self._is_maldi or False

    def get_frame_count(self) -> int:
        """Get the total number of frames."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM Frames")
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Error getting frame count: {e}")
            return len(self._coordinates)

    def get_coverage_stats(self) -> Dict[str, int]:
        """Get statistics about coordinate cache coverage."""
        total_frames = self.get_frame_count()
        cached_frames = len(self._coordinates)

        return {
            "total_frames": total_frames,
            "cached_frames": cached_frames,
            "coverage_percent": (cached_frames / max(1, total_frames)) * 100,
            "loaded_ranges": len(self._loaded_ranges),
        }

    def clear_cache(self) -> None:
        """Clear all cached coordinates."""
        self._coordinates.clear()
        self._loaded_ranges.clear()
        self._dimensions = None
        logger.info("Cleared coordinate cache")

    def optimize_cache(self, keep_recent: int = 1000) -> None:
        """
        Optimize cache by keeping only recently accessed coordinates.

        Args:
            keep_recent: Number of recent coordinates to keep
        """
        if len(self._coordinates) <= keep_recent:
            return

        # Keep the most recent coordinates (by frame ID)
        sorted_frames = sorted(self._coordinates.keys())
        frames_to_keep = sorted_frames[-keep_recent:]

        new_coordinates = {
            frame_id: self._coordinates[frame_id] for frame_id in frames_to_keep
        }

        removed_count = len(self._coordinates) - len(new_coordinates)
        self._coordinates = new_coordinates

        # Clear loaded ranges as they may no longer be valid
        self._loaded_ranges.clear()

        logger.info(
            f"Optimized cache: removed {removed_count} coordinates, kept {len(new_coordinates)}"
        )
