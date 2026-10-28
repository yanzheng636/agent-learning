"""统一配置入口：新增环境变量时，在 Settings 中添加对应的类型字段。"""

from dotenv import find_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=find_dotenv() or None,
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        case_sensitive=False,
    )

    # 字段名不区分大小写：llm_model 对应 .env 中的 LLM_MODEL。
    # 默认留空，允许无模型示例导入；创建模型客户端时再检查必要配置。
    llm_api_key: str = Field(default="", repr=False)
    llm_model: str = ""
    llm_base_url: str = ""


settings = Settings()
