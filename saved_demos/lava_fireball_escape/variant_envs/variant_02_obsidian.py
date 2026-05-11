from demo_envs import DemoLavaFallingFireEnv as _BaseVariantEnv


class DemoLavaFallingFireEnvVariant(_BaseVariantEnv):
    pass


GENERATED_ENV_CLASS = "DemoLavaFallingFireEnvVariant"
SOURCE_PROMPT = DemoLavaFallingFireEnvVariant.prompt
