from fastapi import Request, HTTPException
import time

# Simple in-memory limiter (per IP)
REQUEST_LIMIT = 30     # requests
TIME_WINDOW = 60       # seconds

clients = {}

def rate_limiter(request: Request):
    ip = request.client.host
    now = time.time()

    if ip not in clients:
        clients[ip] = []

    # remove old requests
    clients[ip] = [t for t in clients[ip] if now - t < TIME_WINDOW]

    if len(clients[ip]) >= REQUEST_LIMIT:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please slow down."
        )

    clients[ip].append(now)