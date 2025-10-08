#!/bin/bash

# RAG知识库问答系统 - Nginx设置脚本
# 用于初始化和管理Nginx配置

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查是否为root用户
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_warning "检测到root用户，建议使用普通用户运行此脚本"
    fi
}

# 检查Docker是否安装
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker未安装，请先安装Docker"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! command -v docker compose &> /dev/null; then
        log_error "Docker Compose未安装，请先安装Docker Compose"
        exit 1
    fi
    
    log_success "Docker环境检查通过"
}

# 创建必要的目录
create_directories() {
    log_info "创建必要的目录..."
    
    # 创建Nginx相关目录
    mkdir -p docker/nginx/conf.d
    mkdir -p docker/nginx/ssl
    mkdir -p logs/nginx
    mkdir -p data/nginx/cache
    mkdir -p data/www/static
    
    # 设置权限
    chmod 755 docker/nginx/conf.d
    chmod 755 logs/nginx
    chmod 755 data/nginx/cache
    
    log_success "目录创建完成"
}

# 验证Nginx配置
validate_nginx_config() {
    log_info "验证Nginx配置..."
    
    # 使用Docker临时容器验证配置
    if docker run --rm -v "$(pwd)/docker/nginx/nginx.conf:/etc/nginx/nginx.conf:ro" \
                       -v "$(pwd)/docker/nginx/conf.d:/etc/nginx/conf.d:ro" \
                       nginx:1.25-alpine nginx -t; then
        log_success "Nginx配置验证通过"
        return 0
    else
        log_error "Nginx配置验证失败"
        return 1
    fi
}

# 生成SSL证书 (自签名，用于测试)
generate_ssl_cert() {
    log_info "生成自签名SSL证书..."
    
    SSL_DIR="docker/nginx/ssl"
    
    if [[ ! -f "$SSL_DIR/privkey.pem" ]] || [[ ! -f "$SSL_DIR/fullchain.pem" ]]; then
        # 生成私钥
        openssl genrsa -out "$SSL_DIR/privkey.pem" 2048
        
        # 生成证书签名请求
        openssl req -new -key "$SSL_DIR/privkey.pem" -out "$SSL_DIR/cert.csr" \
            -subj "/C=CN/ST=Beijing/L=Beijing/O=RAG System/OU=IT Department/CN=localhost"
        
        # 生成自签名证书
        openssl x509 -req -days 365 -in "$SSL_DIR/cert.csr" \
            -signkey "$SSL_DIR/privkey.pem" -out "$SSL_DIR/fullchain.pem"
        
        # 清理临时文件
        rm "$SSL_DIR/cert.csr"
        
        # 设置权限
        chmod 600 "$SSL_DIR/privkey.pem"
        chmod 644 "$SSL_DIR/fullchain.pem"
        
        log_success "SSL证书生成完成"
    else
        log_info "SSL证书已存在，跳过生成"
    fi
}

# 启动Nginx服务
start_nginx() {
    log_info "启动Nginx服务..."
    
    # 检查是否有运行中的Nginx容器
    if docker ps -q -f name=rag_nginx_local | grep -q .; then
        log_warning "检测到运行中的Nginx容器，正在重启..."
        docker restart rag_nginx_local
    else
        # 使用docker-compose启动
        if [[ -f "docker-compose.local.yml" ]]; then
            docker compose -f docker-compose.local.yml up -d nginx
        else
            log_error "未找到docker-compose.local.yml文件"
            return 1
        fi
    fi
    
    # 等待服务启动
    sleep 5
    
    # 检查服务状态
    if docker ps -q -f name=rag_nginx_local | grep -q .; then
        log_success "Nginx服务启动成功"
        
        # 显示服务信息
        echo ""
        log_info "服务访问地址:"
        echo "  HTTP:  http://localhost"
        echo "  HTTPS: https://localhost (如果启用SSL)"
        echo "  API文档: http://localhost/docs"
        echo "  健康检查: http://localhost/health"
        echo ""
    else
        log_error "Nginx服务启动失败"
        return 1
    fi
}

# 停止Nginx服务
stop_nginx() {
    log_info "停止Nginx服务..."
    
    if docker ps -q -f name=rag_nginx_local | grep -q .; then
        docker stop rag_nginx_local
        log_success "Nginx服务已停止"
    else
        log_warning "未找到运行中的Nginx容器"
    fi
}

# 重新加载Nginx配置
reload_nginx() {
    log_info "重新加载Nginx配置..."
    
    if docker ps -q -f name=rag_nginx_local | grep -q .; then
        # 先验证配置
        if validate_nginx_config; then
            docker exec rag_nginx_local nginx -s reload
            log_success "Nginx配置重新加载成功"
        else
            log_error "配置验证失败，取消重新加载"
            return 1
        fi
    else
        log_error "未找到运行中的Nginx容器"
        return 1
    fi
}

# 查看Nginx日志
view_logs() {
    log_info "查看Nginx日志..."
    
    if docker ps -q -f name=rag_nginx_local | grep -q .; then
        echo ""
        log_info "最近的访问日志:"
        docker exec rag_nginx_local tail -n 20 /var/log/nginx/access.log
        
        echo ""
        log_info "最近的错误日志:"
        docker exec rag_nginx_local tail -n 20 /var/log/nginx/error.log
    else
        log_error "未找到运行中的Nginx容器"
        return 1
    fi
}

# 测试负载均衡
test_load_balancing() {
    log_info "测试负载均衡..."
    
    # 检查健康检查端点
    if curl -f -s http://localhost/health > /dev/null; then
        log_success "健康检查端点正常"
    else
        log_error "健康检查端点异常"
    fi
    
    # 检查API端点
    if curl -f -s http://localhost/api/system/health > /dev/null; then
        log_success "API端点正常"
    else
        log_warning "API端点可能未启动或异常"
    fi
    
    # 检查文档端点
    if curl -f -s http://localhost/docs > /dev/null; then
        log_success "文档端点正常"
    else
        log_warning "文档端点可能未启动或异常"
    fi
}

# 显示帮助信息
show_help() {
    echo "RAG知识库问答系统 - Nginx设置脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  setup     - 初始化设置 (创建目录、验证配置)"
    echo "  ssl       - 生成SSL证书"
    echo "  start     - 启动Nginx服务"
    echo "  stop      - 停止Nginx服务"
    echo "  restart   - 重启Nginx服务"
    echo "  reload    - 重新加载配置"
    echo "  logs      - 查看日志"
    echo "  test      - 测试负载均衡"
    echo "  validate  - 验证配置"
    echo "  help      - 显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 setup    # 初始化设置"
    echo "  $0 start    # 启动服务"
    echo "  $0 test     # 测试服务"
}

# 主函数
main() {
    case "${1:-help}" in
        setup)
            check_root
            check_docker
            create_directories
            validate_nginx_config
            log_success "Nginx设置完成"
            ;;
        ssl)
            generate_ssl_cert
            ;;
        start)
            check_docker
            validate_nginx_config && start_nginx
            ;;
        stop)
            stop_nginx
            ;;
        restart)
            stop_nginx
            sleep 2
            validate_nginx_config && start_nginx
            ;;
        reload)
            reload_nginx
            ;;
        logs)
            view_logs
            ;;
        test)
            test_load_balancing
            ;;
        validate)
            validate_nginx_config
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"