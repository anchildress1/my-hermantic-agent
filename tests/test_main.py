"""Tests for application entrypoint signal handling."""

import signal
from unittest.mock import MagicMock, patch

import src.main as main_module


def test_install_signal_handlers_registers_sigterm(monkeypatch):
    """SIGTERM handler should be registered."""
    mock_signal = MagicMock()
    monkeypatch.setattr(main_module.signal, "signal", mock_signal)

    main_module._install_signal_handlers()

    assert mock_signal.call_count == 1
    assert mock_signal.call_args[0][0] == signal.SIGTERM
    assert callable(mock_signal.call_args[0][1])


def test_install_signal_handlers_handler_raises_keyboard_interrupt(monkeypatch):
    """Registered SIGTERM handler should raise KeyboardInterrupt."""
    captured = {}

    def fake_signal(sig, handler):
        captured[sig] = handler

    monkeypatch.setattr(main_module.signal, "signal", fake_signal)
    main_module._install_signal_handlers()

    with patch.object(main_module.logger, "warning"):
        try:
            captured[signal.SIGTERM](signal.SIGTERM, None)
            raised = False
        except KeyboardInterrupt:
            raised = True

    assert raised is True


def test_main_invokes_signal_handler_setup(monkeypatch):
    """Main should install signal handlers before config loading."""
    install_called = {"value": False}

    def fake_install():
        install_called["value"] = True

    monkeypatch.setattr(main_module, "_install_signal_handlers", fake_install)
    monkeypatch.setattr(main_module, "setup_logging", lambda: None)
    monkeypatch.setattr(
        main_module, "get_settings", MagicMock(side_effect=Exception("bad env"))
    )

    rc = main_module.main()

    assert install_called["value"] is True
    assert rc == 1
