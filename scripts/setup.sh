#!/bin/bash

# RAG知识库问答系统 - 项目初始化脚本
# 用于快速设置开发环境和依赖

set -e  # 遇到错误立即退出

echo "🚀 RAG知识库问答系统 - 项目初始化"
echo "=================================="

# 检查Python版本
echo "📋 检查Python环境..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "当前Python版本: $python_version"

# 检查是否满足最低版本要求 (3.9+)
required_version="3.9"
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
    echo "❌ 错误: 需要Python 3.9或更高版本"
    exit 1
fi
echo "✅ Python版本检查通过"

# 检查Docker环境
echo "📋 检查Docker环境..."
if ! command -v docker &> /dev/null; then
    echo "❌ 错误: 未找到Docker，请先安装Docker"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ 错误: 未找到docker-compose，请先安装docker-compose"
    exit 1
fi
echo "✅ Docker环境检查通过"

# 创建虚拟环境
echo "📦 创建Python虚拟环境..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ 虚拟环境创建完成"
else
    echo "✅ 虚拟环境已存在"
fi

# 激活虚拟环境
echo "🔧 激活虚拟环境..."
source venv/bin/activate

# 升级pip
echo "📦 升级pip..."
pip install --upgrade pip

# 安装依赖
echo "📦 安装Python依赖包..."
pip install -r requirements.txt

# 安装开发依赖
echo "📦 安装开发依赖包..."
pip install -e ".[dev]"

# 创建环境变量文件
echo "⚙️  配置环境变量..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✅ 已创建.env文件，请根据实际环境修改配置"
else
    echo "✅ .env文件已存在"
fi

# 创建必要的目录
echo "📁 创建必要的目录..."
mkdir -p data/uploads
mkdir -p data/documents
mkdir -p data/chroma
mkdir -p data/redis
mkdir -p logs
echo "✅ 目录创建完成"

# 设置权限
echo "🔐 设置文件权限..."
chmod +x scripts/*.sh
echo "✅ 权限设置完成"

# 运行代码质量检查
echo "🔍 运行代码质量检查..."
echo "运行Black格式化..."
black src/ tests/ --check --diff || true

echo "运行isort导入排序..."
isort src/ tests/ --check-only --diff || true

echo "运行flake8代码检查..."
flake8 src/ tests/ || true

echo "运行mypy类型检查..."
mypy src/ || true

# 运行测试
echo "🧪 运行测试..."
pytest tests/ -v --cov=src --cov-report=term-missing || true

echo ""
echo "🎉 项目初始化完成！"
echo "=================================="
echo "📝 下一步操作:"
echo "1. 修改 .env 文件中的配置项"
echo "2. 运行 'docker-compose up -d' 启动服务"
echo "3. 运行 'python -m src.main' 启动应用"
echo ""
echo "🔧 常用命令:"
echo "  启动服务: docker-compose up -d"
echo "  停止服务: docker-compose down"
echo "  查看日志: docker-compose logs -f"
echo "  运行测试: pytest"
echo "  代码格式化: black src/ tests/"
echo "  类型检查: mypy src/"
echo ""
echo "📚 更多信息请查看 README.md"