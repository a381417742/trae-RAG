#!/bin/bash

# RAG知识库问答系统 - GitHub上传脚本
# 使用说明：请先在GitHub创建仓库，然后修改下面的仓库URL

echo "🚀 开始上传RAG知识库问答系统到GitHub..."

# 配置变量（请修改为您的GitHub仓库信息）
GITHUB_USERNAME="a381417742"  # 替换为您的GitHub用户名
REPO_NAME="trae-RAG"  # 仓库名称
REPO_URL="https://github.com/a381417742/trae-RAG.git"

echo "📋 仓库信息："
echo "   用户名: ${GITHUB_USERNAME}"
echo "   仓库名: ${REPO_NAME}"
echo "   仓库URL: ${REPO_URL}"
echo ""

# 检查是否已经是Git仓库
if [ ! -d ".git" ]; then
    echo "🔧 初始化Git仓库..."
    git init
    echo "✅ Git仓库初始化完成"
else
    echo "ℹ️  Git仓库已存在"
fi

# 配置Git用户信息（如果尚未配置）
echo "🔧 配置Git用户信息..."
read -p "请输入您的Git用户名: " git_username
read -p "请输入您的Git邮箱: " git_email

git config user.name "$git_username"
git config user.email "$git_email"
echo "✅ Git用户信息配置完成"

# 添加所有文件到暂存区
echo "📁 添加文件到Git暂存区..."
git add .
echo "✅ 文件添加完成"

# 检查暂存区状态
echo "📊 检查暂存区状态..."
git status

# 提交更改
echo "💾 提交更改..."
git commit -m "🎉 Initial commit: RAG知识库问答系统

✨ 功能特性:
- 🤖 智能问答系统基于Ollama大语言模型
- 📚 支持多格式文档处理(PDF/TXT/MD/DOCX)
- 🔍 向量检索基于Chroma向量数据库
- ⚡ Redis缓存提升响应速度
- 🛡️ 完整的安全防护机制
- 📊 性能监控和指标统计
- 🐳 Docker容器化部署
- 🧪 完整的测试套件
- 📖 详细的系统文档

🏗️ 技术栈:
- Backend: FastAPI + Python 3.11+
- AI/ML: Ollama + BGE Embedding
- Database: Chroma Vector DB + Redis Cache
- Deploy: Docker + Docker Compose + Nginx
- Monitor: Prometheus + Grafana
- Test: Pytest + Coverage

📦 项目结构完整，包含源码、测试、文档、配置等所有必要文件"

echo "✅ 代码提交完成"

# 添加远程仓库
echo "🔗 添加远程仓库..."
git remote add origin "$REPO_URL"
echo "✅ 远程仓库添加完成"

# 推送到GitHub
echo "⬆️  推送代码到GitHub..."
echo "注意：如果是首次推送，可能需要输入GitHub用户名和密码/Token"
git branch -M main
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 恭喜！RAG知识库问答系统已成功上传到GitHub！"
    echo "🔗 仓库地址: ${REPO_URL}"
    echo ""
    echo "📋 后续操作建议："
    echo "   1. 在GitHub仓库页面完善README.md"
    echo "   2. 设置仓库标签和主题"
    echo "   3. 配置GitHub Pages展示文档"
    echo "   4. 设置CI/CD自动化流程"
    echo ""
else
    echo "❌ 上传失败，请检查："
    echo "   1. GitHub仓库URL是否正确"
    echo "   2. 网络连接是否正常"
    echo "   3. GitHub认证信息是否正确"
    echo "   4. 仓库是否已存在且有权限"
fi