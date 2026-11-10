# Implemented models

## Thermal networks

::: pysip.statespace.thermal_network
    options:
      inherited_members: false
      filters: ["!^_", "!^set_constant_continuous_ssm$", "!^update_continuous_ssm$"]

## Gaussian processes

::: pysip.statespace.gaussian_process
    options:
      inherited_members: false
      filters: ["!^_", "!^set_constant_continuous_ssm$", "!^update_continuous_ssm$"]

## Latent-force models

::: pysip.statespace.latent_force_model.LatentForceModel

::: pysip.statespace.latent_force_model.R2C2_Qgh_Matern32
