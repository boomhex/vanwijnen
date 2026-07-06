import posixpath
from collections.abc import Collection

from fastapi import Request
from fastapi.responses import PlainTextResponse, RedirectResponse
from nicegui import app
from starlette.middleware.base import BaseHTTPMiddleware

from services import auth

UNRESTRICTED_ROUTES = {'/login', '/favicon.ico'}


def storage_request_allowed(path: str, workspaces: Collection[str]) -> bool:
    """Allow /storage requests only inside workspaces the user may open.

    Any `..` in a storage path is rejected, and the path is also normalized
    so `/storage/team1/../team2/x` cannot slip past a prefix check; the
    static file server resolves those segments the same way.
    """
    raw_segments = [segment for segment in path.split('/') if segment]
    if not raw_segments or raw_segments[0] != 'storage':
        return True
    if '..' in raw_segments:
        return False
    segments = [segment for segment in posixpath.normpath(path).split('/') if segment]
    return len(segments) >= 2 and segments[0] == 'storage' and segments[1] in workspaces


class AuthMiddleware(BaseHTTPMiddleware):
    """Redirect unauthenticated requests to the login page.

    Runs for every HTTP request, so it also protects the ``/storage``
    static files (offer PDFs and extraction results), not just the pages.
    Authenticated users can only fetch storage files from workspaces they
    were granted (``/storage/<workspace>/...``). NiceGUI internals
    (``/_nicegui...``) stay reachable because the login page needs them.
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if not app.storage.user.get('authenticated', False):
            if not path.startswith('/_nicegui') and path not in UNRESTRICTED_ROUTES:
                app.storage.user['referrer_path'] = path
                return RedirectResponse('/login')
        elif path.startswith('/storage'):
            granted = auth.user_workspaces(app.storage.user.get('username'))
            if not storage_request_allowed(path, granted):
                return PlainTextResponse('Forbidden', status_code=403)
        return await call_next(request)
