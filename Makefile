# RAG知识库问答系统 - Makefile
# 提供常用的项目管理和部署命令

.PHONY: help setup install clean test lint format build up down logs status health

# 默认目标
help:
	@echo "RAG知识库问答系统 - 可用命令:"
	@echo ""
	@echo "🚀 项目初始化:"
	@echo "  setup     - 初始化项目环境"
	@echo "  install   - 安装Python依赖"
	@echo ""
	@echo "🔧 开发工具:"
	@echo "  test      - 运行测试"
	@echo "  lint      - 代码检查"
	@echo "  format    - 代码格式化"
	@echo "  clean     - 清理临时文件"
	@echo ""
	@echo "🐳 Docker操作:"
	@echo "  build     - 构建Docker镜像"
	@echo "  up        - 启动所有服务"
	@echo "  down      - 停止所有服务"
	@echo "  logs      - 查看服务日志"
	@echo "  status    - 查看服务状态"
	@echo "  health    - 健康检查"
	@echo ""

# 项目初始化
setup:
	@echo "🚀 初始化项目环境..."
	chmod +x scripts/*.sh
	./scripts/setup.sh

# 安装依赖
install:
	@echo "📦 安装Python依赖..."
	pip install -r requirements.txt
	pip install -e ".[dev]"

# 清理临时文件
clean:
	@echo "🧹 清理临时文件..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

# 运行测试
test:
	@echo "🧪 运行测试..."
	pytest tests/ -v --cov=src --cov-report=term-missing

# 代码检查
lint:
	@echo "🔍 运行代码检查..."
	flake8 src/ tests/
	mypy src/
	isort src/ tests/ --check-only --diff

# 代码格式化
format:
	@echo "✨ 格式化代码..."
	black src/ tests/
	isort src/ tests/

# 构建Docker镜像
build:
	@echo "🐳 构建Docker镜像..."
	docker-compose build

# 启动所有服务
up:
	@echo "🚀 启动所有服务..."
	docker-compose up -d
	@echo "✅ 服务启动完成"
	@echo "📊 访问地址:"
	@echo "  - API文档: http://localhost/docs"
	@echo "  - Grafana: http://localhost:3000 (admin/admin123)"
	@echo "  - Prometheus: http://localhost:9090"

# 停止所有服务
down:
	@echo "🛑 停止所有服务..."
	docker-compose down

# 查看服务日志
logs:
	@echo "📋 查看服务日志..."
	docker-compose logs -f

# 查看服务状态
status:
	@echo "📊 查看服务状态..."
	docker-compose ps

# 健康检查
health:
	@echo "🏥 执行健康检查..."
	@echo "检查Nginx..."
	@curl -f http://localhost/health || echo "❌ Nginx不健康"
	@echo "检查FastAPI..."
	@curl -f http://localhost:8000/health || echo "❌ FastAPI不健康"
	@echo "检查Ollama..."
	@curl -f http://localhost:11434/api/tags || echo "❌ Ollama不健康"
	@echo "检查Chroma..."
	@curl -f http://localhost:8002/api/v1/heartbeat || echo "❌ Chroma不健康"
	@echo "检查Redis..."
	@docker exec rag_redis redis-cli ping || echo "❌ Redis不健康"

# 开发模式启动
dev:
	@echo "🔧 启动开发模式..."
	uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# 生产模式启动
prod:
	@echo "🚀 启动生产模式..."
	uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4

# 数据库迁移 (如果需要)
migrate:
	@echo "🗄️ 执行数据库迁移..."
	# 这里可以添加数据库迁移命令

# 备份数据
backup:
	@echo "💾 备份数据..."
	mkdir -p backups/$(shell date +%Y%m%d_%H%M%S)
	cp -r data/ backups/$(shell date +%Y%m%d_%H%M%S)/

# 恢复数据
restore:
	@echo "🔄 恢复数据..."
	@echo "请指定备份目录: make restore BACKUP_DIR=backups/20231201_120000"
	@if [ -z "$(BACKUP_DIR)" ]; then echo "❌ 请指定BACKUP_DIR参数"; exit 1; fi
	cp -r $(BACKUP_DIR)/data/ ./

# 监控
monitor:
	@echo "📊 打开监控面板..."
	@echo "Grafana: http://localhost:3000"
	@echo "Prometheus: http://localhost:9090"
	@echo "API文档: http://localhost/docs"