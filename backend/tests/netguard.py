"""Make any attempt to reach a non-loopback address fail the test run.

Installed for the whole backend suite by the autouse fixtures in conftest.py. A connect / sendto to an address that is
not loopback, or a DNS lookup of a name that is not `localhost`, raises `OutboundNetworkBlocked` AND is recorded; the
per-test fixture fails the test when anything was recorded, so even code that swallows the exception cannot hide an
outbound call. Unix sockets and loopback traffic (the TestClient, fake local servers) are untouched.
"""

from __future__ import annotations

import ipaddress
import socket

violations: list[str] = []


class OutboundNetworkBlocked(AssertionError):
    """The code under test tried to talk to a non-loopback address."""


def _is_loopback_host(host) -> bool:
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    if host in (None, ""):
        return True  # wildcard bind / local lookups
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.split("%")[0]).is_loopback
    except ValueError:
        return False


def _check_address(address, what: str) -> None:
    if isinstance(address, str | bytes):  # AF_UNIX path
        return
    host = address[0] if isinstance(address, tuple | list) and address else address
    if not _is_loopback_host(host):
        msg = f"{what} to non-loopback address {host!r} blocked: the app must never go online"
        violations.append(msg)
        raise OutboundNetworkBlocked(msg)


def install(monkeypatch) -> None:
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_sendto = socket.socket.sendto
    real_getaddrinfo = socket.getaddrinfo

    def connect(self, address):
        if self.family != socket.AF_UNIX:
            _check_address(address, "connect")
        return real_connect(self, address)

    def connect_ex(self, address):
        if self.family != socket.AF_UNIX:
            _check_address(address, "connect_ex")
        return real_connect_ex(self, address)

    def sendto(self, data, *args):
        if self.family != socket.AF_UNIX and args:
            _check_address(args[-1], "sendto")
        return real_sendto(self, data, *args)

    def getaddrinfo(host, *args, **kwargs):
        if not _is_loopback_host(host):
            msg = f"DNS lookup of {host!r} blocked: the app must never go online"
            violations.append(msg)
            raise OutboundNetworkBlocked(msg)
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket.socket, "sendto", sendto)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
