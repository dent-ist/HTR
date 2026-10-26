"""The suite-wide guard against outbound network calls (tests/netguard.py) and the facts that make it meaningful."""

import socket

import pytest

from tests import netguard


@pytest.fixture
def expect_violations():
    """These tests provoke the guard on purpose; clear its record so the per-test check does not fail them."""
    netguard.violations.clear()
    yield netguard.violations
    netguard.violations.clear()


@pytest.mark.parametrize("host", ["203.0.113.7", "8.8.8.8", "93.184.216.34", "2001:4860:4860::8888"])
def test_connect_to_a_public_address_is_blocked(host, expect_violations):
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    s = socket.socket(family, socket.SOCK_STREAM)
    s.settimeout(0.2)
    try:
        with pytest.raises(netguard.OutboundNetworkBlocked):
            s.connect((host, 443))
        with pytest.raises(netguard.OutboundNetworkBlocked):
            s.connect_ex((host, 443))
    finally:
        s.close()
    assert len(expect_violations) == 2


def test_dns_lookup_of_a_public_name_is_blocked(expect_violations):
    with pytest.raises(netguard.OutboundNetworkBlocked):
        socket.getaddrinfo("example.com", 443)
    with pytest.raises(netguard.OutboundNetworkBlocked):
        socket.create_connection(("pypi.org", 443), timeout=0.2)
    assert len(expect_violations) == 2


def test_udp_to_a_public_address_is_blocked(expect_violations):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        with pytest.raises(netguard.OutboundNetworkBlocked):
            s.sendto(b"x", ("8.8.8.8", 53))
    finally:
        s.close()
    assert expect_violations


def test_loopback_still_works():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    client = socket.create_connection(server.getsockname(), timeout=2)
    conn, _ = server.accept()
    client.sendall(b"ping")
    assert conn.recv(4) == b"ping"
    for s in (client, conn, server):
        s.close()
    assert socket.getaddrinfo("localhost", 80)  # localhost lookups are fine


def test_a_swallowed_violation_is_still_recorded(expect_violations):
    """Code that catches the exception cannot hide the call: the violation list is what fails the test."""
    s = socket.socket()
    try:
        try:
            s.connect(("203.0.113.9", 80))
        except Exception:  # noqa: BLE001, S110  (the point of this test)
            pass
    finally:
        s.close()
    assert expect_violations and "203.0.113.9" in expect_violations[0]
