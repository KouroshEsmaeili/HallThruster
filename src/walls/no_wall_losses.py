class WallLossModel:
    pass


class NoWallLosses(WallLossModel):
    pass


def freq_electron_wall(model: NoWallLosses, _params, _i) -> float:
    return 0.0


def wall_electron_current(model: NoWallLosses, _params, _i) -> float:
    return 0.0


def wall_power_loss(Q: 'np.ndarray', model: NoWallLosses, _params) -> None:
    Q[:] = 0.0


def wall_loss_scale(model: NoWallLosses) -> float:
    return 0.0
