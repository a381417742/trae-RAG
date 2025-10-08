# 🤝 贡献指南

感谢您对RAG知识库问答系统的关注！我们欢迎所有形式的贡献，包括但不限于：

- 🐛 Bug报告和修复
- ✨ 新功能建议和实现
- 📚 文档改进
- 🧪 测试用例添加
- 🎨 UI/UX改进
- 🔧 性能优化

## 📋 开发环境搭建

### 1. 克隆项目
```bash
git clone https://github.com/your-username/RAG-Knowledge-QA-System.git
cd RAG-Knowledge-QA-System
```

### 2. 创建虚拟环境
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # 开发依赖
```

### 4. 配置环境
```bash
cp .env.example .env
# 编辑 .env 文件配置开发环境参数
```

### 5. 启动开发服务
```bash
# 启动外部依赖服务
docker-compose -f docker-compose.local.yml up -d

# 启动应用
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

## 🔄 开发流程

### 1. 创建Issue
在开始开发前，请先创建或查找相关的Issue，描述您要解决的问题或实现的功能。

### 2. Fork项目
点击项目页面右上角的"Fork"按钮，将项目Fork到您的GitHub账户。

### 3. 创建分支
```bash
git checkout -b feature/your-feature-name
# 或
git checkout -b bugfix/issue-number
```

### 4. 开发代码
- 遵循项目的代码规范
- 添加必要的注释和文档
- 编写相应的测试用例

### 5. 运行测试
```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_specific.py

# 生成覆盖率报告
pytest --cov=src --cov-report=html
```

### 6. 提交代码
```bash
git add .
git commit -m "feat: add new feature description"
# 或
git commit -m "fix: resolve issue #123"
```

### 7. 推送分支
```bash
git push origin feature/your-feature-name
```

### 8. 创建Pull Request
在GitHub上创建Pull Request，详细描述您的更改。

## 📝 代码规范

### Python代码规范
- 遵循 [PEP 8](https://www.python.org/dev/peps/pep-0008/) 代码风格
- 使用 `black` 进行代码格式化
- 使用 `flake8` 进行代码检查
- 使用 `mypy` 进行类型检查

```bash
# 格式化代码
black src/ tests/

# 检查代码风格
flake8 src/ tests/

# 类型检查
mypy src/
```

### 提交信息规范
使用 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

类型说明：
- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更新
- `style`: 代码格式调整
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建过程或辅助工具的变动

示例：
```
feat(api): add batch question processing endpoint

Add new endpoint for processing multiple questions simultaneously
to improve system throughput and user experience.

Closes #123
```

## 🧪 测试指南

### 测试类型
- **单元测试**: 测试单个函数或类的功能
- **集成测试**: 测试模块间的交互
- **性能测试**: 测试系统性能和负载能力
- **安全测试**: 测试安全漏洞和防护机制

### 测试编写规范
```python
import pytest
from unittest.mock import Mock, patch

class TestDocumentProcessor:
    """文档处理器测试类"""
    
    def test_process_pdf_success(self):
        """测试PDF文档处理成功场景"""
        # Arrange
        processor = DocumentProcessor()
        mock_file = Mock()
        
        # Act
        result = processor.process_pdf(mock_file)
        
        # Assert
        assert result is not None
        assert len(result.chunks) > 0
    
    def test_process_invalid_file_format(self):
        """测试无效文件格式处理"""
        processor = DocumentProcessor()
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            processor.process_file("invalid.xyz")
```

### 测试覆盖率要求
- 新增代码的测试覆盖率应达到 80% 以上
- 核心业务逻辑的测试覆盖率应达到 90% 以上
- 关键安全功能必须有完整的测试覆盖

## 📚 文档规范

### 代码文档
```python
def process_question(
    self, 
    question: str, 
    max_results: int = 5,
    similarity_threshold: float = 0.7
) -> QuestionResult:
    """
    处理用户问题并生成答案
    
    Args:
        question: 用户问题文本
        max_results: 最大检索结果数量
        similarity_threshold: 相似度阈值
        
    Returns:
        QuestionResult: 包含答案和相关文档的结果对象
        
    Raises:
        ValueError: 当问题为空或过长时
        ProcessingError: 当处理过程中发生错误时
        
    Example:
        >>> processor = QAProcessor()
        >>> result = processor.process_question("什么是AI？")
        >>> print(result.answer)
    """
```

### API文档
使用FastAPI的自动文档生成功能，确保所有API端点都有完整的文档：

```python
@router.post("/ask", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    qa_processor: QAProcessor = Depends(get_qa_processor)
) -> QuestionResponse:
    """
    处理单个问题
    
    - **question**: 用户问题 (必填)
    - **max_results**: 最大检索结果数 (可选，默认5)
    - **similarity_threshold**: 相似度阈值 (可选，默认0.7)
    
    返回包含答案、相关文档和置信度的响应。
    """
```

## 🐛 Bug报告

### Bug报告模板
```markdown
## Bug描述
简要描述遇到的问题

## 复现步骤
1. 执行操作A
2. 执行操作B
3. 观察到错误

## 期望行为
描述您期望发生的情况

## 实际行为
描述实际发生的情况

## 环境信息
- 操作系统: [e.g. Ubuntu 20.04]
- Python版本: [e.g. 3.11.0]
- 项目版本: [e.g. v2.0.0]

## 附加信息
- 错误日志
- 截图
- 其他相关信息
```

## ✨ 功能建议

### 功能建议模板
```markdown
## 功能描述
简要描述建议的新功能

## 使用场景
描述这个功能的使用场景和目标用户

## 详细设计
- 功能详细说明
- API设计 (如适用)
- UI设计 (如适用)

## 实现考虑
- 技术实现方案
- 性能影响
- 兼容性考虑

## 替代方案
描述其他可能的实现方式
```

## 🔍 代码审查

### 审查清单
- [ ] 代码符合项目规范
- [ ] 功能实现正确完整
- [ ] 测试覆盖充分
- [ ] 文档更新及时
- [ ] 性能影响可接受
- [ ] 安全考虑充分
- [ ] 向后兼容性

### 审查流程
1. 自动化检查通过 (CI/CD)
2. 至少一位维护者审查
3. 解决所有审查意见
4. 合并到主分支

## 🏷️ 发布流程

### 版本号规范
使用 [Semantic Versioning](https://semver.org/) 规范：
- `MAJOR.MINOR.PATCH`
- `MAJOR`: 不兼容的API更改
- `MINOR`: 向后兼容的功能添加
- `PATCH`: 向后兼容的Bug修复

### 发布步骤
1. 更新版本号
2. 更新CHANGELOG.md
3. 创建发布标签
4. 构建和发布Docker镜像
5. 发布GitHub Release

## 📞 获取帮助

如果您在贡献过程中遇到任何问题，可以通过以下方式获取帮助：

- 📧 发送邮件到项目维护者
- 💬 在GitHub Discussions中提问
- 🐛 在GitHub Issues中报告问题
- 📱 加入项目交流群

## 🙏 致谢

感谢所有为项目做出贡献的开发者！您的每一个贡献都让这个项目变得更好。

---

再次感谢您的贡献！🎉