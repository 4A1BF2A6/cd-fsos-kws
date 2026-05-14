def compose(transforms):
    """Compose callables left-to-right, matching torchnet.transform.compose."""

    def _composed(value):
        for transform in transforms:
            value = transform(value)
        return value

    return _composed
