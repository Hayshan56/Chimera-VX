#!/usr/bin/env python3
# chimera-vx/server/main_server.py
# The Heart of the Ultimate CTF

import asyncio
import aiohttp
from aiohttp import web
import json
import hashlib
import secrets
import base64
import time
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import traceback

# Local imports
from database import Database
from package_generator import PackageGenerator
from verification import VerificationEngine
from anti_cheat import AntiCheatSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ChimeraServer:
    """Main server class for Chimera-VX CTF"""
    
    def __init__(self, config_path: str = "config/server_config.json"):
        self.config = self.load_config(config_path)
        self.db = Database(self.config['database']['path'])
        self.generator = PackageGenerator(self.config)
        self.verifier = VerificationEngine(self.config)
        self.anti_cheat = AntiCheatSystem(self.config)
        
        # Server state
        self.active_sessions: Dict[str, Dict] = {}
        self.puzzle_cache: Dict[str, bytes] = {}
        self.rate_limits: Dict[str, List[float]] = {}
        
        # Statistics
        self.stats = {
            'players_registered': 0,
            'puzzles_generated': 0,
            'solutions_submitted': 0,
            'flags_captured': 0,
            'cheat_attempts': 0
        }
        
        logger.info("Chimera-VX Server initialized")
        
    def load_config(self, config_path: str) -> Dict:
        """Load server configuration"""
        default_config = {
            'server': {
                'host': '0.0.0.0',
                'port': 8080,
                'debug': False,
                'secret_key': secrets.token_hex(32),
                'session_timeout': 86400,  # 24 hours
                'max_players': 1000
            },
            'database': {
                'path': 'data/chimera.db',
                'backup_interval': 3600  # 1 hour
            },
            'security': {
                'require_proof_of_work': True,
                'pow_difficulty': 5,  # Leading zeros
                'max_attempts_per_puzzle': 10,
                'puzzle_timeout': 86400,  # 24 hours
                'hardware_verification': True,
                'rate_limit_window': 60,  # seconds
                'rate_limit_max': 100  # requests per window
            },
            'puzzles': {
                'total_circles': 12,
                'min_solve_time': 72,  # hours
                'max_solve_time': 168,  # hours
                'puzzle_order': [
                    'quantum', 'dna', 'radio', 'fpga', 'minecraft',
                    'usb', 'temporal', 'cryptographic', 'hardware',
                    'forensic', 'network', 'meta'
                ]
            },
            'paths': {
                'data_dir': 'data',
                'puzzle_dir': 'puzzles',
                'log_dir': 'logs',
                'temp_dir': 'temp',
                'key_dir': 'keys'
            }
        }
        
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                # Merge with defaults
                for key in default_config:
                    if key in user_config:
                        default_config[key].update(user_config[key])
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            
        return default_config
    
    async def start(self):
        """Start the server"""
        app = web.Application()
        
        # Setup middleware
        app.middlewares.append(self.rate_limit_middleware)
        app.middlewares.append(self.error_handler_middleware)
        app.middlewares.append(self.security_headers_middleware)
        
        # Register routes
        app.router.add_post('/api/v1/register', self.handle_register)
        app.router.add_post('/api/v1/login', self.handle_login)
        app.router.add_post('/api/v1/logout', self.handle_logout)
        app.router.add_get('/api/v1/status', self.handle_status)
        app.router.add_get('/api/v1/profile', self.handle_profile)
        app.router.add_get('/api/v1/challenge', self.handle_challenge)
        app.router.add_post('/api/v1/submit', self.handle_submit)
        app.router.add_get('/api/v1/progress', self.handle_progress)
        app.router.add_get('/api/v1/leaderboard', self.handle_leaderboard)
        app.router.add_post('/api/v1/reset', self.handle_reset)
        app.router.add_post('/api/v1/verify/hardware', self.handle_hardware_verify)
        
        # Static files (for web interface)
        app.router.add_static('/static/', 'static')
        
        # WebSocket for real-time updates
        app.router.add_get('/ws', self.handle_websocket)
        
        # Start background tasks
        asyncio.create_task(self.cleanup_tasks())
        asyncio.create_task(self.backup_database())
        asyncio.create_task(self.monitor_system())
        
        # Start server
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(
            runner,
            self.config['server']['host'],
            self.config['server']['port']
        )
        
        await site.start()
        
        logger.info(f"Server started on http://{self.config['server']['host']}:{self.config['server']['port']}")
        logger.info(f"API Documentation: http://{self.config['server']['host']}:{self.config['server']['port']}/docs")
        
        # Keep server running
        await asyncio.Event().wait()
    
    # ==================== MIDDLEWARE ====================
    
   @web.middleware
    async def rate_limit_middleware(self, request: web.Request, handler):
        """Rate limiting middleware"""
        client_ip = request.remote
        
        # Check rate limit
        if not self.check_rate_limit(client_ip):
            return web.json_response(
                {'error': 'Rate limit exceeded'},
                status=429
            )
        
        return await handler(request)
    
    @web.middleware
    async def error_handler_middleware(self, request: web.Request, handler):
        """Global error handling middleware"""
        try:
            return await handler(request)
        except web.HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unhandled error: {e}\n{traceback.format_exc()}")
            return web.json_response(
                {'error': 'Internal server error'},
                status=500
            )
    
    @web.middleware
    async def security_headers_middleware(self, request: web.Request, handler):
        """Add security headers to all responses"""
        response = await handler(request)
        
        # Security headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        response.headers['Content-Security-Policy'] = "default-src 'self'"
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
        
        return response
    
  # ==================== HANDLERS ====================
    
    async def handle_register(self, request: web.Request) -> web.Response:
        """Handle player registration"""
        try:
            data = await request.json()
            
            # Validate input
            required_fields = ['username', 'email', 'hardware_fingerprint']
            for field in required_fields:
                if field not in data:
                    return web.json_response(
                        {'error': f'Missing required field: {field}'},
                        status=400
                    )
            
            # Check if Proof of Work is required
            if self.config['security']['require_proof_of_work']:
                if 'proof_of_work' not in data:
                    return web.json_response(
                        {'error': 'Proof of Work required'},
                        status=400
                    )
                
                # Verify PoW
                if not self.verify_proof_of_work(data['proof_of_work']):
                    return web.json_response(
                        {'error': 'Invalid Proof of Work'},
                        status=400
                    )
            
            # Check if username is available
            if self.db.get_player_by_username(data['username']):
                return web.json_response(
                    {'error': 'Username already exists'},
                    status=409
                )
            
            # Check if hardware fingerprint is unique
            if self.config['security']['hardware_verification']:
                existing = self.db.get_player_by_hardware(data['hardware_fingerprint'])
                if existing:
                    return web.json_response(
                        {'error': 'Hardware already registered'},
                        status=409
                    )
            
            # Create player
            player_id = self.db.create_player(
                username=data['username'],
                email=data['email'],
                hardware_fingerprint=data['hardware_fingerprint']
            )
            
            # Generate session token
            session_token = secrets.token_hex(32)
            session_hash = hashlib.sha256(session_token.encode()).hexdigest()
            
            # Store session
            self.db.create_session(
                player_id=player_id,
                session_hash=session_hash,
                ip_address=request.remote
            )
            
            # Generate initial challenge package
            package = await self.generator.generate_initial_package(player_id)
            
            # Update statistics
            self.stats['players_registered'] += 1
            
            logger.info(f"New player registered: {data['username']} (ID: {player_id})")
            
            return web.json_response({
                'player_id': player_id,
                'session_token': session_token,
                'message': 'Registration successful',
                'initial_package': package,
                'total_circles': self.config['puzzles']['total_circles']
            })
            
        except json.JSONDecodeError:
            return web.json_response(
                {'error': 'Invalid JSON'},
                status=400
            )
    
    async def handle_login(self, request: web.Request) -> web.Response:
        """Handle player login"""
        try:
            data = await request.json()
            
            # Validate input
            if 'username' not in data or 'hardware_fingerprint' not in data:
                return web.json_response(
                    {'error': 'Missing username or hardware fingerprint'},
                    status=400
                )
            
            # Get player
            player = self.db.get_player_by_username(data['username'])
            if not player:
                return web.json_response(
                    {'error': 'Invalid username'},
                    status=401
                )
            
            # Verify hardware fingerprint
            if self.config['security']['hardware_verification']:
                if player['hardware_fingerprint'] != data['hardware_fingerprint']:
                    logger.warning(f"Hardware mismatch for player {data['username']}")
                    self.anti_cheat.log_suspicious_activity(
                        player['id'],
                        'hardware_mismatch',
                        request.remote
                    )
                    return web.json_response(
                        {'error': 'Hardware verification failed'},
                        status=401
                    )
            
            # Generate new session token
            session_token = secrets.token_hex(32)
            session_hash = hashlib.sha256(session_token.encode()).hexdigest()
            
            # Store session
            self.db.create_session(
                player_id=player['id'],
                session_hash=session_hash,
                ip_address=request.remote
            )
            
            logger.info(f"Player logged in: {data['username']}")
            
            return web.json_response({
                'player_id': player['id'],
                'session_token': session_token,
                'progress': player['progress'],
                'total_time': player['total_time'],
                'current_circle': player['current_circle']
            })
            
        except json.JSONDecodeError:
            return web.json_response(
                {'error': 'Invalid JSON'},
                status=400
            )
    
    async def handle_logout(self, request: web.Request) -> web.Response:
        """Handle player logout"""
        session_token = request.headers.get('X-Session-Token')
        if not session_token:
            return web.json_response(
                {'error': 'Session token required'},
                status=401
            )
        
        session_hash = hashlib.sha256(session_token.encode()).hexdigest()
        self.db.delete_session(session_hash)
        
        return web.json_response({
            'message': 'Logout successful'
        })
    
    async def handle_status(self, request: web.Request) -> web.Response:
        """Get server status"""
        return web.json_response({
            'status': 'online',
            'version': '1.0.0',
            'players_online': len(self.active_sessions),
            'players_registered': self.stats['players_registered'],
            'puzzles_generated': self.stats['puzzles_generated'],
            'flags_captured': self.stats['flags_captured'],
            'uptime': time.time() - self.start_time,
            'server_time': datetime.utcnow().isoformat()
        })
    
    async def handle_profile(self, request: web.Request) -> web.Response:
        """Get player profile"""
        player = await self.authenticate_player(request)
        if not player:
            return web.json_response(
                {'error': 'Authentication required'},
                status=401
            )
        
        return web.json_response({
            'player_id': player['id'],
            'username': player['username'],
            'progress': player['progress'],
            'total_time': player['total_time'],
            'current_circle': player['current_circle'],
            'registration_date': player['created_at'],
            'last_active': player['last_active'],
            'achievements': json.loads(player.get('achievements', '[]'))
        })
    
    async def handle_challenge(self, request: web.Request) -> web.Response:
        """Get current challenge for player"""
        player = await self.authenticate_player(request)
        if not player:
            return web.json_response(
                {'error': 'Authentication required'},
                status=401
            )
        
        # Check if player is rate limited for challenges
        if not self.can_request_challenge(player['id']):
            return web.json_response(
                {'error': 'Challenge request too frequent'},
                status=429
            )
        # Get current puzzle
        puzzle = self.db.get_current_puzzle(player['id'])
        if not puzzle:
            # Generate new puzzle
            puzzle_data = await self.generator.generate_puzzle(
                player_id=player['id'],
                circle_number=player['current_circle'],
                player_data=player
            )
            
            # Store puzzle
            puzzle_id = self.db.create_puzzle(
                player_id=player['id'],
                circle_number=player['current_circle'],
                puzzle_type=self.config['puzzles']['puzzle_order'][player['current_circle'] - 1],
                puzzle_data=json.dumps(puzzle_data['puzzle']),
                solution_hash=puzzle_data['solution_hash'],
                created_at=int(time.time())
            )
            
            puzzle = {
                'id': puzzle_id,
                'puzzle_data': puzzle_data['puzzle'],
                'created_at': int(time.time()),
                'circle': player['current_circle'],
                'type': self.config['puzzles']['puzzle_order'][player['current_circle'] - 1]
            }
            
            self.stats['puzzles_generated'] += 1
        
        # Check if puzzle expired
        puzzle_age = time.time() - puzzle['created_at']
        if puzzle_age > self.config['security']['puzzle_timeout']:
            # Regenerate puzzle
            puzzle_data = await self.generator.generate_puzzle(
                player_id=player['id'],
                circle_number=player['current_circle'],
                player_data=player
            )
            
            self.db.update_puzzle(
                puzzle_id=puzzle['id'],
                puzzle_data=json.dumps(puzzle_data['puzzle']),
                solution_hash=puzzle_data['solution_hash'],
                created_at=int(time.time())
            )
            
            puzzle['puzzle_data'] = puzzle_data['puzzle']
            puzzle['created_at'] = int(time.time())
        
        # Add anti-cheat watermark
        watermarked_puzzle = self.anti_cheat.add_watermark(
            puzzle['puzzle_data'],
            player['id'],
            request.remote
        )
        
        return web.json_response({
            'puzzle_id': puzzle['id'],
            'circle': puzzle['circle'],
            'type': puzzle['type'],
            'created_at': puzzle['created_at'],
            'time_remaining': self.config['security']['puzzle_timeout'] - puzzle_age,
            'puzzle': watermarked_puzzle,
            'hints_available': 3,  # Maximum hints per puzzle
            'attempts_remaining': self.config['security']['max_attempts_per_puzzle'] - puzzle.get('attempts', 0)
        })
    
    async def handle_submit(self, request: web.Request) -> web.Response:
        """Handle solution submission"""
        player = await self.authenticate_player(request)
        if not player:
            return web.json_response(
                {'error': 'Authentication required'},
                status=401
            )
        
        try:
            data = await request.json()
            
            # Validate input
            if 'puzzle_id' not in data or 'solution' not in data:
                return web.json_response(
                    {'error': 'Missing puzzle_id or solution'},
                    status=400
                )
            
            # Get puzzle
            puzzle = self.db.get_puzzle(data['puzzle_id'])
            if not puzzle or puzzle['player_id'] != player['id']:
                return web.json_response(
                    {'error': 'Invalid puzzle'},
                    status=404
                )
            
            # Check if puzzle is expired
            puzzle_age = time.time() - puzzle['created_at']
            if puzzle_age > self.config['security']['puzzle_timeout']:
                return web.json_response(
                    {'error': 'Puzzle expired'},
                    status=410
                )
            
            # Check attempts limit
            if puzzle['attempts'] >= self.config['security']['max_attempts_per_puzzle']:
                return web.json_response(
                    {'error': 'Maximum attempts exceeded'},
                    status=429
                )
            
            # Verify solution
            is_correct, verification_data = await self.verifier.verify_solution(
                puzzle_data=json.loads(puzzle['puzzle_data']),
                solution=data['solution'],
                puzzle_type=puzzle['type'],
                player_id=player['id']
            )
            
            # Run anti-cheat checks
            cheat_detected, cheat_data = await self.anti_cheat.check_submission(
                player_id=player['id'],
                puzzle_id=puzzle['id'],
                solution=data['solution'],
                verification_data=verification_data,
                ip_address=request.remote
            )
            
            if cheat_detected:
                logger.warning(f"Cheat detected for player {player['username']}")
                self.stats['cheat_attempts'] += 1
                
                # Apply penalty
                penalty = self.anti_cheat.apply_penalty(player['id'], cheat_data)
                
                return web.json_response({
                    'correct': False,
                    'cheat_detected': True,
                    'penalty': penalty,
                    'message': 'Anti-cheat system triggered'
                })
            
            # Update puzzle attempts
            self.db.increment_puzzle_attempts(puzzle['id'])
            
            # Update statistics
            self.stats['solutions_submitted'] += 1
            
            if is_correct:
                # Mark puzzle as solved
                self.db.mark_puzzle_solved(
                    puzzle_id=puzzle['id'],
                    solved_at=int(time.time()),
                    solution=data['solution']
                )
                
                # Update player progress
                self.db.update_player_progress(
                    player_id=player['id'],
                    new_circle=player['current_circle'] + 1,
                    time_spent=verification_data.get('solve_time', 0)
                )
                
                # Check if player completed all circles
                if player['current_circle'] + 1 > self.config['puzzles']['total_circles']:
                    # Generate final flag
                    final_flag = await self.generator.generate_final_flag(player['id'])
                    
                    # Record completion
                    self.db.record_completion(
                        player_id=player['id'],
                        final_flag=final_flag,
                        total_time=player['total_time'] + verification_data.get('solve_time', 0)
                    )
                    
                    # Update statistics
                    self.stats['flags_captured'] += 1
                    
                    logger.info(f"Player {player['username']} completed all circles!")
                    
                    return web.json_response({
                        'correct': True,
                        'circle_completed': True,
                        'all_circles_completed': True,
                        'final_flag': final_flag,
                        'completion_time': datetime.utcnow().isoformat(),
                        'rank': self.db.get_player_rank(player['id'])
                    })
                else:
                    return web.json_response({
                        'correct': True,
                        'circle_completed': True,
                        'next_circle': player['current_circle'] + 1,
                        'message': f'Circle {player["current_circle"]} completed!'
                    })
            else:
                return web.json_response({
                    'correct': False,
                    'attempts_remaining': self.config['security']['max_attempts_per_puzzle'] - puzzle['attempts'] - 1,
                    'hint': verification_data.get('hint', '')
                })
                
        except json.JSONDecodeError:
            return web.json_response(
                {'error': 'Invalid JSON'},
                status=400
            )
    
    async def handle_progress(self, request: web.Request) -> web.Response:
        """Get player progress"""
        player = await self.authenticate_player(request)
        if not player:
            return web.json_response(
                {'error': 'Authentication required'},
                status=401
            )
        
        # Get all solved puzzles
        solved_puzzles = self.db.get_solved_puzzles(player['id'])
        
        # Calculate statistics
        total_time = player['total_time']
        average_time = total_time / len(solved_puzzles) if solved_puzzles else 0
        
        return web.json_response({
            'player_id': player['id'],
            'current_circle': player['current_circle'],
            'total_circles': self.config['puzzles']['total_circles'],
            'solved_puzzles': len(solved_puzzles),
            'total_time': total_time,
            'average_time_per_puzzle': average_time,
            'puzzle_breakdown': [
                {
                    'circle': p['circle_number'],
                    'type': p['type'],
                    'solved_at': p['solved_at'],
                    'attempts': p['attempts'],
                    'time_spent': p['time_spent']
                }
                for p in solved_puzzles
            ]
        })
    
    async def handle_leaderboard(self, request: web.Request) -> web.Response:
        """Get global leaderboard"""
        # Get query parameters
        limit = int(request.query.get('limit', 100))
        offset = int(request.query.get('offset', 0))
        
        # Get leaderboard
        leaderboard = self.db.get_leaderboard(limit, offset)
        
        # Format response
        formatted = []
        for i, entry in enumerate(leaderboard):
            formatted.append({
                'rank': offset + i + 1,
                'player_id': entry['id'],
                'username': entry['username'],
                'progress': entry['progress'],
                'total_time': entry['total_time'],
                'completed_at': entry.get('completed_at'),
                'last_active': entry['last_active']
            })
        
        return web.json_response({
            'leaderboard': formatted,
            'total_players': self.db.get_total_players(),
            'limit': limit,
            'offset': offset
        })
    
    async def handle_reset(self, request: web.Request) -> web.Response:
        """Reset player progress (with confirmation)"""
        player = await self.authenticate_player(request)
        if not player:
            return web.json_response(
                {'error': 'Authentication required'},
                status=401
            )
        
        try:
            data = await request.json()
            
            # Require confirmation
            if 'confirm' not in data or data['confirm'] != 'I UNDERSTAND THIS WILL DELETE ALL MY PROGRESS':
                return web.json_response({
                    'warning': 'This action will delete ALL your progress',
                    'confirmation_required': 'Type: I UNDERSTAND THIS WILL DELETE ALL MY PROGRESS'
                })
            
            # Reset player
            self.db.reset_player(player['id'])
            
            logger.info(f"Player {player['username']} reset their progress")
            
            return web.json_response({
                'message': 'Progress reset successfully',
                'new_player_id': player['id']  # Same ID, fresh start
            })
            
        except json.JSONDecodeError:
            return web.json_response(
                {'error': 'Invalid JSON'},
                status=400
            )
    
    async def handle_hardware_verify(self, request: web.Request) -> web.Response:
        """Verify hardware fingerprint"""
        player = await self.authenticate_player(request)
        if not player:
            return web.json_response(
                {'error': 'Authentication required'},
                status=401
            )
        
        try:
            data = await request.json()
            
            if 'hardware_data' not in data:
                return web.json_response(
                    {'error': 'Hardware data required'},
                    status=400
                )
            
            # Verify hardware
            is_valid, verification_data = self.anti_cheat.verify_hardware(
                player_id=player['id'],
                current_fingerprint=player['hardware_fingerprint'],
                new_data=data['hardware_data']
            )
            
            if not is_valid:
                return web.json_response({
                    'verified': False,
                    'reason': verification_data.get('reason', 'Hardware mismatch'),
                    'suspicious': verification_data.get('suspicious', False)
                })
            
            return web.json_response({
                'verified': True,
                'hardware_id': verification_data.get('hardware_id'),
                'next_verification': time.time() + 86400  # 24 hours
            })
            
        except json.JSONDecodeError:
            return web.json_response(
                {'error': 'Invalid JSON'},
                status=400
            )
    
    async def handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """Handle WebSocket connections for real-time updates"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        player = None
        session_token = request.query.get('token')
        
        if session_token:
            session_hash = hashlib.sha256(session_token.encode()).hexdigest()
            player = self.db.get_player_by_session(session_hash)
        
        if not player:
            await ws.close(code=1008, message='Authentication required')
            return ws
        
        # Register connection
        connection_id = secrets.token_hex(16)
        self.active_sessions[connection_id] = {
            'player_id': player['id'],
            'websocket': ws,
            'connected_at': time.time(),
            'last_ping': time.time()
        }
        
        try:
            # Send initial data
            await ws.send_json({
                'type': 'welcome',
                'player_id': player['id'],
                'connection_id': connection_id,
                'server_time': datetime.utcnow().isoformat()
            })
            
            # Handle messages
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self.handle_websocket_message(connection_id, data)
                    except json.JSONDecodeError:
                        await ws.send_json({
                            'type': 'error',
                            'message': 'Invalid JSON'
                        })
                elif msg.type == web.WSMsgType.ERROR:
                    logger.error(f'WebSocket error: {ws.exception()}')
                    
        finally:
            # Remove connection
            if connection_id in self.active_sessions:
                del self.active_sessions[connection_id]
        
        return ws
    
    async def handle_websocket_message(self, connection_id: str, data: Dict):
        """Handle WebSocket messages"""
        if connection_id not in self.active_sessions:
            return
        
        ws = self.active_sessions[connection_id]['websocket']
        player_id = self.active_sessions[connection_id]['player_id']
        
        message_type = data.get('type')
        
        if message_type == 'ping':
            self.active_sessions[connection_id]['last_ping'] = time.time()
            await ws.send_json({
                'type': 'pong',
                'timestamp': time.time()
            })
            
        elif message_type == 'progress_update':
            # Send progress update to client
            player = self.db.get_player(player_id)
            if player:
                await ws.send_json({
                    'type': 'progress',
                    'current_circle': player['current_circle'],
                    'total_time': player['total_time']
                })

     # ==================== HELPER METHODS ====================
    
    async def authenticate_player(self, request: web.Request) -> Optional[Dict]:
        """Authenticate player from request"""
        session_token = request.headers.get('X-Session-Token')
        if not session_token:
            return None
        
        session_hash = hashlib.sha256(session_token.encode()).hexdigest()
        player = self.db.get_player_by_session(session_hash)
        
        if player:
            # Update last active
            self.db.update_player_last_active(player['id'])
        
        return player
    
    def verify_proof_of_work(self, pow_data: str) -> bool:
        """Verify proof of work"""
        difficulty = self.config['security']['pow_difficulty']
        target = '0' * difficulty
        
        # Simple hash-based PoW
        h = hashlib.sha256(pow_data.encode()).hexdigest()
        return h.startswith(target)
    
    def check_rate_limit(self, client_ip: str) -> bool:
        """Check if client is rate limited"""
        window = self.config['security']['rate_limit_window']
        max_requests = self.config['security']['rate_limit_max']
        
        now = time.time()
        
        if client_ip not in self.rate_limits:
            self.rate_limits[client_ip] = []
        
        # Clean old requests
        self.rate_limits[client_ip] = [
            t for t in self.rate_limits[client_ip]
            if now - t < window
        ]
        
        # Check if limit exceeded
        if len(self.rate_limits[client_ip]) >= max_requests:
            return False
        
        # Add current request
        self.rate_limits[client_ip].append(now)
        return True
    
    def can_request_challenge(self, player_id: int) -> bool:
        """Check if player can request a new challenge"""
        # Minimum 60 seconds between challenge requests
        last_request = self.db.get_last_challenge_request(player_id)
        if last_request and time.time() - last_request < 60:
            return False
        return True

  # ==================== BACKGROUND TASKS ====================
    
    async def cleanup_tasks(self):
        """Run cleanup tasks periodically"""
        while True:
            try:
                # Clean old sessions
                self.db.clean_old_sessions(self.config['server']['session_timeout'])
                
                # Clean rate limits
                window = self.config['security']['rate_limit_window']
                now = time.time()
                for ip in list(self.rate_limits.keys()):
                    self.rate_limits[ip] = [
                        t for t in self.rate_limits[ip]
                        if now - t < window
                    ]
                    if not self.rate_limits[ip]:
                        del self.rate_limits[ip]
                
                # Clean WebSocket connections
                for conn_id in list(self.active_sessions.keys()):
                    last_ping = self.active_sessions[conn_id]['last_ping']
                    if time.time() - last_ping > 300:  # 5 minutes
                        try:
                            await self.active_sessions[conn_id]['websocket'].close()
                        except:
                            pass
                        del self.active_sessions[conn_id]
                
                # Log cleanup
                logger.debug("Cleanup tasks completed")
                
            except Exception as e:
                logger.error(f"Error in cleanup tasks: {e}")
            
            await asyncio.sleep(300)  # Run every 5 minutes
    
    async def backup_database(self):
        """Backup database periodically"""
        while True:
            try:
                backup_path = f"backups/chimera_backup_{int(time.time())}.db"
                self.db.backup(backup_path)
                logger.info(f"Database backed up to {backup_path}")
            except Exception as e:
                logger.error(f"Error backing up database: {e}")
            
            await asyncio.sleep(self.config['database']['backup_interval'])
    
    async def monitor_system(self):
        """Monitor system health"""
        while True:
            try:
                # Check disk space
                import shutil
                total, used, free = shutil.disk_usage("/")
                if free / total < 0.1:  # Less than 10% free
                    logger.warning(f"Low disk space: {free / total:.1%} free")
                
                # Check memory
                import psutil
                memory = psutil.virtual_memory()
                if memory.percent > 90:
                    logger.warning(f"High memory usage: {memory.percent}%")
                
                # Log statistics
                logger.info(f"Active sessions: {len(self.active_sessions)}")
                logger.info(f"Rate limited IPs: {len(self.rate_limits)}")
                
            except Exception as e:
                logger.error(f"Error in system monitor: {e}")
            
            await asyncio.sleep(60)  # Run every minute

   # ==================== MAIN ====================

async def main():
    """Main entry point"""
    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("backups").mkdir(exist_ok=True)
    Path("temp").mkdir(exist_ok=True)
    
    # Initialize and start server
    server = ChimeraServer()
    server.start_time = time.time()
    
    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
    except Exception as e:
        logger.error(f"Server crashed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
```

      
        
        