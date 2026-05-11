from types import SimpleNamespace

import pytest

import setup_env


def test_system_info_normalizes_x86_alias(monkeypatch):
    monkeypatch.setattr(setup_env.platform, "system", lambda: "Linux")
    monkeypatch.setattr(setup_env.platform, "machine", lambda: "AMD64")

    assert setup_env.system_info() == ("Linux", "x86_64")


def test_system_info_normalizes_arm_alias(monkeypatch):
    monkeypatch.setattr(setup_env.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(setup_env.platform, "machine", lambda: "aarch64")

    assert setup_env.system_info() == ("Darwin", "arm64")


def test_get_model_name_uses_supported_hf_repo(monkeypatch):
    monkeypatch.setattr(
        setup_env,
        "args",
        SimpleNamespace(
            hf_repo="HF1BitLLM/Llama3-8B-1.58-100B-tokens",
            model_dir="models/ignored",
        ),
        raising=False,
    )

    assert setup_env.get_model_name() == "Llama3-8B-1.58-100B-tokens"


def test_get_model_name_uses_normalized_model_directory(monkeypatch):
    monkeypatch.setattr(
        setup_env,
        "args",
        SimpleNamespace(hf_repo=None, model_dir="models/local-bitnet/"),
        raising=False,
    )

    assert setup_env.get_model_name() == "local-bitnet"


def test_parse_args_accepts_arch_specific_quant_type(monkeypatch):
    monkeypatch.setattr(setup_env.platform, "system", lambda: "Linux")
    monkeypatch.setattr(setup_env.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        setup_env.sys,
        "argv",
        ["setup_env.py", "--model-dir", "models/local-bitnet", "--quant-type", "tl2"],
    )

    args = setup_env.parse_args()

    assert args.model_dir == "models/local-bitnet"
    assert args.quant_type == "tl2"


def test_parse_args_rejects_quant_type_for_other_arch(monkeypatch):
    monkeypatch.setattr(setup_env.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(setup_env.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        setup_env.sys,
        "argv",
        ["setup_env.py", "--model-dir", "models/local-bitnet", "--quant-type", "tl2"],
    )

    with pytest.raises(SystemExit):
        setup_env.parse_args()
