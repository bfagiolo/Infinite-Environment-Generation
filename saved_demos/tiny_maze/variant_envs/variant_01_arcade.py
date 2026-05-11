from demo_envs import DemoTinyMazeEnv as _BaseVariantEnv


class DemoTinyMazeEnvVariant(_BaseVariantEnv):
    pass


GENERATED_ENV_CLASS = "DemoTinyMazeEnvVariant"
SOURCE_PROMPT = DemoTinyMazeEnvVariant.prompt
