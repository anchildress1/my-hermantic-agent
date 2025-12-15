from src.interfaces.cli.chat import print_help


def test_print_help_with_memory_store(capsys):
    class Dummy:
        pass

    print_help(Dummy())
    out = capsys.readouterr().out
    assert "Memory Commands" in out
