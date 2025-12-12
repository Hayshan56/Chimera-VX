#!/usr/bin/env python3
# chimera-vx/server/database.py
# Database management for Chimera-VX

import sqlite3
import json
import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class Database:
    """Database manager for Chimera-VX"""
    
    def __init__(self, db_path: str = "data/chimera.db"):
        self.db_path = db_path
        self.init_database()
        
    def get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_database(self):
        """Initialize database schema"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Players table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS players (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT NOT NULL,
                    hardware_fingerprint TEXT NOT NULL,
                    progress INTEGER DEFAULT 0,
                    current_circle INTEGER DEFAULT 1,
                    total_time INTEGER DEFAULT 0,
                    created_at INTEGER NOT NULL,
                    last_active INTEGER,
                    achievements TEXT DEFAULT '[]',
                    metadata TEXT DEFAULT '{}',
                    is_active BOOLEAN DEFAULT 1,
                    UNIQUE(username),
                    UNIQUE(hardware_fingerprint)
                )
            ''')
            
            # Sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id INTEGER NOT NULL,
                    session_hash TEXT UNIQUE NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT,
                    created_at INTEGER NOT NULL,
                    last_used INTEGER,
                    expires_at INTEGER NOT NULL,
                    FOREIGN KEY (player_id) REFERENCES players(id) ON DELETE CASCADE
                )
            ''')
            
            # Puzzles table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS puzzles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id INTEGER NOT NULL,
                    circle_number INTEGER NOT NULL,
                    type TEXT NOT NULL,
                    puzzle_data TEXT NOT NULL,
                    solution_hash TEXT NOT NULL,
                    attempts INTEGER DEFAULT 0,
                    created_at INTEGER NOT NULL,
                    solved_at INTEGER,
                    solution TEXT,
                    time_spent INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'active',
                    metadata TEXT DEFAULT '{}',
                    FOREIGN KEY (player_id) REFERENCES players(id) ON DELETE CASCADE
                )
            ''')
            
            # Submissions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS submissions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id INTEGER NOT NULL,
                    puzzle_id INTEGER NOT NULL,
                    submission TEXT NOT NULL,
                    is_correct BOOLEAN NOT NULL,
                    submitted_at INTEGER NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT,
                    verification_data TEXT DEFAULT '{}',
                    cheat_detected BOOLEAN DEFAULT 0,
                    FOREIGN KEY (player_id) REFERENCES players(id) ON DELETE CASCADE,
                    FOREIGN KEY (puzzle_id) REFERENCES puzzles(id) ON DELETE CASCADE
                )
            ''')
            
            # Hardware profiles table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS hardware_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id INTEGER NOT NULL,
                    profile_data TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    last_verified INTEGER,
                    consistency_score REAL DEFAULT 1.0,
                    suspicious_count INTEGER DEFAULT 0,
                    FOREIGN KEY (player_id) REFERENCES players(id) ON DELETE CASCADE
                )
            ''')
            
            # Anti-cheat logs
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cheat_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    severity INTEGER DEFAULT 1,
                    details TEXT DEFAULT '{}',
                    detected_at INTEGER NOT NULL,
                    action_taken TEXT,
                    FOREIGN KEY (player_id) REFERENCES players(id) ON DELETE CASCADE
                )
            ''')
            
            # Leaderboard cache
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS leaderboard (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id INTEGER UNIQUE NOT NULL,
                    username TEXT NOT NULL,
                    progress INTEGER DEFAULT 0,
                    total_time INTEGER DEFAULT 0,
                    completed_at INTEGER,
                    last_updated INTEGER NOT NULL,
                    FOREIGN KEY (player_id) REFERENCES players(id) ON DELETE CASCADE
                )
            ''')
            
            # Statistics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric TEXT UNIQUE NOT NULL,
                    value INTEGER DEFAULT 0,
                    updated_at INTEGER NOT NULL
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_players_username ON players(username)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_players_hardware ON players(hardware_fingerprint)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_player ON sessions(player_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_hash ON sessions(session_hash)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_puzzles_player ON puzzles(player_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_puzzles_circle ON puzzles(circle_number)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_puzzles_status ON puzzles(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_submissions_player ON submissions(player_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_submissions_puzzle ON submissions(puzzle_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_hardware_player ON hardware_profiles(player_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_cheat_player ON cheat_logs(player_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_leaderboard_progress ON leaderboard(progress DESC, total_time ASC)')
            
            conn.commit()
        
        logger.info("Database initialized")
    
    # ==================== PLAYER METHODS ====================
    
    def create_player(self, username: str, email: str, hardware_fingerprint: str) -> int:
        """Create a new player"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO players (username, email, hardware_fingerprint, created_at, last_active)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                username,
                email,
                hardware_fingerprint,
                int(time.time()),
                int(time.time())
            ))
            
            player_id = cursor.lastrowid
            
            # Create hardware profile
            cursor.execute('''
                INSERT INTO hardware_profiles (player_id, profile_data, created_at, last_verified)
                VALUES (?, ?, ?, ?)
            ''', (
                player_id,
                json.dumps({'fingerprint': hardware_fingerprint}),
                int(time.time()),
                int(time.time())
            ))
            
            conn.commit()
            
            logger.info(f"Created player {username} (ID: {player_id})")
            return player_id
    
    def get_player(self, player_id: int) -> Optional[Dict]:
        """Get player by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM players WHERE id = ?', (player_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_player_by_username(self, username: str) -> Optional[Dict]:
        """Get player by username"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM players WHERE username = ?', (username,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_player_by_hardware(self, hardware_fingerprint: str) -> Optional[Dict]:
        """Get player by hardware fingerprint"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM players WHERE hardware_fingerprint = ?', (hardware_fingerprint,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def update_player_progress(self, player_id: int, new_circle: int, time_spent: int):
        """Update player progress"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE players 
                SET progress = progress + 1,
                    current_circle = ?,
                    total_time = total_time + ?,
                    last_active = ?
                WHERE id = ?
            ''', (new_circle, time_spent, int(time.time()), player_id))
            
            # Update leaderboard cache
            self.update_leaderboard(player_id)
            
            conn.commit()
            
            logger.debug(f"Updated progress for player {player_id}: circle {new_circle}")
    
    def update_player_last_active(self, player_id: int):
        """Update player's last active timestamp"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE players SET last_active = ? WHERE id = ?
            ''', (int(time.time()), player_id))
            conn.commit()
    
    def reset_player(self, player_id: int):
        """Reset player progress"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Reset player
            cursor.execute('''
                UPDATE players 
                SET progress = 0,
                    current_circle = 1,
                    total_time = 0,
                    achievements = '[]',
                    metadata = '{}'
                WHERE id = ?
            ''', (player_id,))
            
            # Delete puzzles
            cursor.execute('DELETE FROM puzzles WHERE player_id = ?', (player_id,))
            
            # Delete submissions
            cursor.execute('DELETE FROM submissions WHERE player_id = ?', (player_id,))
            
            # Delete leaderboard entry
            cursor.execute('DELETE FROM leaderboard WHERE player_id = ?', (player_id,))
            
            conn.commit()
            
            logger.info(f"Reset player {player_id}")
    
    def get_player_rank(self, player_id: int) -> Optional[int]:
        """Get player's rank on leaderboard"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get all players ordered by progress and time
            cursor.execute('''
                SELECT id FROM players 
                WHERE is_active = 1 
                ORDER BY progress DESC, total_time ASC
            ''')
            
            rows = cursor.fetchall()
            for i, row in enumerate(rows, 1):
                if row['id'] == player_id:
                    return i
            
            return None
    
    # ==================== SESSION METHODS ====================
    
    def create_session(self, player_id: int, session_hash: str, ip_address: str = None):
        """Create a new session"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            expires_at = int(time.time()) + 86400  # 24 hours
            
            cursor.execute('''
                INSERT INTO sessions (player_id, session_hash, ip_address, created_at, last_used, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                player_id,
                session_hash,
                ip_address,
                int(time.time()),
                int(time.time()),
                expires_at
            ))
            
            conn.commit()
            
            logger.debug(f"Created session for player {player_id}")
    
    def get_player_by_session(self, session_hash: str) -> Optional[Dict]:
        """Get player by session hash"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT p.* FROM players p
                JOIN sessions s ON p.id = s.player_id
                WHERE s.session_hash = ? 
                AND s.expires_at > ?
                AND p.is_active = 1
            ''', (session_hash, int(time.time())))
            
            row = cursor.fetchone()
            if row:
                # Update session last used
                cursor.execute('''
                    UPDATE sessions SET last_used = ? WHERE session_hash = ?
                ''', (int(time.time()), session_hash))
                conn.commit()
                return dict(row)
            
            return None
    
    def delete_session(self, session_hash: str):
        """Delete a session"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM sessions WHERE session_hash = ?', (session_hash,))
            conn.commit()
            logger.debug(f"Deleted session {session_hash}")
    
    def clean_old_sessions(self, timeout: int):
        """Clean expired sessions"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            expired = int(time.time()) - timeout
            cursor.execute('DELETE FROM sessions WHERE expires_at < ?', (expired,))
            deleted = cursor.rowcount
            conn.commit()
            
            if deleted > 0:
                logger.debug(f"Cleaned {deleted} expired sessions")
    
    # ==================== PUZZLE METHODS ====================
    
    def create_puzzle(self, player_id: int, circle_number: int, puzzle_type: str, 
                     puzzle_data: str, solution_hash: str, created_at: int) -> int:
        """Create a new puzzle for player"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO puzzles (player_id, circle_number, type, puzzle_data, 
                                   solution_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                player_id,
                circle_number,
                puzzle_type,
                puzzle_data,
                solution_hash,
                created_at
            ))
            
            puzzle_id = cursor.lastrowid
            conn.commit()
            
            logger.debug(f"Created puzzle {puzzle_id} for player {player_id}")
            return puzzle_id
    
    def get_current_puzzle(self, player_id: int) -> Optional[Dict]:
        """Get player's current unsolved puzzle"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM puzzles 
                WHERE player_id = ? 
                AND status = 'active'
                ORDER BY created_at DESC
                LIMIT 1
            ''', (player_id,))
            
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_puzzle(self, puzzle_id: int) -> Optional[Dict]:
        """Get puzzle by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM puzzles WHERE id = ?', (puzzle_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def increment_puzzle_attempts(self, puzzle_id: int):
        """Increment puzzle attempts counter"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE puzzles SET attempts = attempts + 1 WHERE id = ?
            ''', (puzzle_id,))
            conn.commit()
    
    def mark_puzzle_solved(self, puzzle_id: int, solved_at: int, solution: str):
        """Mark puzzle as solved"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Calculate time spent
            cursor.execute('SELECT created_at FROM puzzles WHERE id = ?', (puzzle_id,))
            created_at = cursor.fetchone()['created_at']
            time_spent = solved_at - created_at
            
            cursor.execute('''
                UPDATE puzzles 
                SET solved_at = ?,
                    solution = ?,
                    time_spent = ?,
                    status = 'solved'
                WHERE id = ?
            ''', (solved_at, solution, time_spent, puzzle_id))
            
            conn.commit()
            
            logger.debug(f"Marked puzzle {puzzle_id} as solved")
    
    def update_puzzle(self, puzzle_id: int, puzzle_data: str, solution_hash: str, created_at: int):
        """Update puzzle data"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE puzzles 
                SET puzzle_data = ?,
                    solution_hash = ?,
                    created_at = ?,
                    attempts = 0
                WHERE id = ?
            ''', (puzzle_data, solution_hash, created_at, puzzle_id))
            conn.commit()
    
    def get_solved_puzzles(self, player_id: int) -> List[Dict]:
        """Get all solved puzzles for player"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM puzzles 
                WHERE player_id = ? 
                AND status = 'solved'
                ORDER BY circle_number
            ''', (player_id,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_last_challenge_request(self, player_id: int) -> Optional[int]:
        """Get timestamp of last challenge request"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT MAX(created_at) as last_request FROM puzzles 
                WHERE player_id = ?
            ''', (player_id,))
            
            row = cursor.fetchone()
            return row['last_request'] if row['last_request'] else None
    
    # ==================== SUBMISSION METHODS ====================
    
    def create_submission(self, player_id: int, puzzle_id: int, submission: str, 
                         is_correct: bool, ip_address: str = None, 
                         verification_data: Dict = None, cheat_detected: bool = False):
        """Create a submission record"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO submissions (player_id, puzzle_id, submission, is_correct,
                                       submitted_at, ip_address, verification_data, cheat_detected)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                player_id,
                puzzle_id,
                submission,
                is_correct,
                int(time.time()),
                ip_address,
                json.dumps(verification_data or {}),
                cheat_detected
            ))
            
            conn.commit()
            
            logger.debug(f"Created submission for puzzle {puzzle_id}")
    
    def get_submissions(self, player_id: int, limit: int = 100) -> List[Dict]:
        """Get player's submissions"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM submissions 
                WHERE player_id = ? 
                ORDER BY submitted_at DESC
                LIMIT ?
            ''', (player_id, limit))
            
            return [dict(row) for row in cursor.fetchall()]

  # ==================== HARDWARE PROFILE METHODS ====================
    
    def update_hardware_profile(self, player_id: int, profile_data: Dict):
        """Update hardware profile"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE hardware_profiles 
                SET profile_data = ?,
                    last_verified = ?,
                    consistency_score = consistency_score * 0.9 + 0.1
                WHERE player_id = ?
            ''', (
                json.dumps(profile_data),
                int(time.time()),
                player_id
            ))
            
            conn.commit()
    
    def get_hardware_profile(self, player_id: int) -> Optional[Dict]:
        """Get hardware profile"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM hardware_profiles WHERE player_id = ?', (player_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def increment_suspicious_count(self, player_id: int):
        """Increment suspicious activity count"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE hardware_profiles 
                SET suspicious_count = suspicious_count + 1,
                    consistency_score = consistency_score * 0.8
                WHERE player_id = ?
            ''', (player_id,))
            conn.commit()

    # ==================== ANTI-CHEAT METHODS ====================
    
    def log_cheat_event(self, player_id: int, event_type: str, severity: int, 
                       details: Dict, action_taken: str = None):
        """Log cheat detection event"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO cheat_logs (player_id, event_type, severity, details, 
                                      detected_at, action_taken)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                player_id,
                event_type,
                severity,
                json.dumps(details),
                int(time.time()),
                action_taken
            ))
            
            conn.commit()
            
            logger.warning(f"Logged cheat event: {event_type} for player {player_id}")
    
    def get_cheat_logs(self, player_id: int, limit: int = 50) -> List[Dict]:
        """Get cheat logs for player"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM cheat_logs 
                WHERE player_id = ? 
                ORDER BY detected_at DESC
                LIMIT ?
            ''', (player_id, limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    # ==================== LEADERBOARD METHODS ====================
    
    def update_leaderboard(self, player_id: int):
        """Update leaderboard cache for player"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get player data
            cursor.execute('''
                SELECT id, username, progress, total_time FROM players 
                WHERE id = ?
            ''', (player_id,))
            
            player = cursor.fetchone()
            if not player:
                return
            
            # Check if player completed all circles
            cursor.execute('''
                SELECT COUNT(*) as completed FROM puzzles 
                WHERE player_id = ? AND status = 'solved'
            ''', (player_id,))
            
            completed = cursor.fetchone()['completed']
            
            # Update or insert leaderboard entry
            cursor.execute('''
                INSERT OR REPLACE INTO leaderboard (player_id, username, progress, 
                                                   total_time, completed_at, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                player['id'],
                player['username'],
                player['progress'],
                player['total_time'],
                int(time.time()) if completed >= 12 else None,
                int(time.time())
            ))
            
            conn.commit()
    
    def get_leaderboard(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Get leaderboard"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT l.*, p.last_active 
                FROM leaderboard l
                JOIN players p ON l.player_id = p.id
                ORDER BY l.progress DESC, l.total_time ASC, l.completed_at ASC
                LIMIT ? OFFSET ?
            ''', (limit, offset))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_total_players(self) -> int:
        """Get total number of active players"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) as count FROM players WHERE is_active = 1')
            return cursor.fetchone()['count']
    
    # ==================== STATISTICS METHODS ====================
    
    def update_statistic(self, metric: str, value: int):
        """Update statistic"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO statistics (metric, value, updated_at)
                VALUES (?, ?, ?)
            ''', (metric, value, int(time.time())))
            
            conn.commit()
    
    def get_statistic(self, metric: str) -> Optional[int]:
        """Get statistic value"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT value FROM statistics WHERE metric = ?', (metric,))
            row = cursor.fetchone()
            return row['value'] if row else None
    
    def increment_statistic(self, metric: str, amount: int = 1):
        """Increment statistic"""
        current = self.get_statistic(metric) or 0
        self.update_statistic(metric, current + amount)
    
    # ==================== UTILITY METHODS ====================
    
    def backup(self, backup_path: str):
        """Create database backup"""
        import shutil
        shutil.copy2(self.db_path, backup_path)
        logger.info(f"Database backed up to {backup_path}")
    
    def vacuum(self):
        """Optimize database"""
        with self.get_connection() as conn:
            conn.execute('VACUUM')
            conn.commit()
            logger.info("Database vacuumed")
    
    def get_database_size(self) -> int:
        """Get database file size in bytes"""
        return Path(self.db_path).stat().st_size if Path(self.db_path).exists() else 0
    
    def record_completion(self, player_id: int, final_flag: str, total_time: int):
        """Record player completion"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Add final flag to achievements
            cursor.execute('SELECT achievements FROM players WHERE id = ?', (player_id,))
            achievements = json.loads(cursor.fetchone()['achievements'])
            achievements.append({
                'type': 'completion',
                'flag': final_flag,
                'completed_at': int(time.time()),
                'total_time': total_time
            })
            
            cursor.execute('''
                UPDATE players 
                SET achievements = ?,
                    current_circle = 13,  # Completed state
                    last_active = ?
                WHERE id = ?
            ''', (json.dumps(achievements), int(time.time()), player_id))
            
            conn.commit()
            
            logger.info(f"Recorded completion for player {player_id}")

# Test the database
if __name__ == "__main__":
    db = Database("test.db")
    print("Database test completed")
```
