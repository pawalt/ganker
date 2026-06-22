from dataclasses import asdict

from ganker.contracts import AdamParams, Datum, ModelInput, SamplingParams, TensorData


def test_contracts_are_plain_picklable_dataclasses():
    datum = Datum(
        model_input=ModelInput.from_ints(tokens=[1, 2, 3]),
        loss_fn_inputs={
            "target_tokens": TensorData.from_ints([2, 3, 4]),
            "weights": TensorData.from_floats([0.0, 1.0, 1.0]),
        },
    )
    params = SamplingParams(max_tokens=8, temperature=0.7)
    adam = AdamParams(learning_rate=1e-4)

    assert asdict(datum)["model_input"]["token_ids"] == [1, 2, 3]
    assert asdict(datum)["loss_fn_inputs"]["target_tokens"]["values"] == [2, 3, 4]
    assert datum.loss_fn_inputs["weights"].tolist() == [0.0, 1.0, 1.0]
    assert params.max_tokens == 8
    assert adam.learning_rate == 1e-4
