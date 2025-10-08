"""
安全测试
验证RAG系统的安全性，包括输入验证、权限控制、数据保护等
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
import io

from src.main import app


class TestInputValidationSecurity:
    """输入验证安全测试"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_sql_injection_prevention(self, client):
        """测试SQL注入防护"""
        # 尝试SQL注入攻击
        malicious_questions = [
            "'; DROP TABLE documents; --",
            "' OR '1'='1",
            "'; DELETE FROM users; --",
            "' UNION SELECT * FROM admin; --"
        ]
        
        for question in malicious_questions:
            response = client.post(
                "/api/qa/ask",
                json={"question": question}
            )
            
            # 系统应该正常处理，不应该执行SQL命令
            assert response.status_code in [200, 422, 500]  # 不应该是数据库错误
            
            if response.status_code == 200:
                data = response.json()
                # 答案不应该包含数据库错误信息
                assert "SQL" not in data.get("answer", "").upper()
                assert "DROP" not in data.get("answer", "").upper()
                assert "DELETE" not in data.get("answer", "").upper()
    
    def test_xss_prevention(self, client):
        """测试XSS攻击防护"""
        # 尝试XSS攻击
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "';alert('XSS');//"
        ]
        
        for payload in xss_payloads:
            response = client.post(
                "/api/qa/ask",
                json={"question": payload}
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "")
                
                # 响应不应该包含未转义的脚本标签
                assert "<script>" not in answer
                assert "javascript:" not in answer
                assert "onerror=" not in answer
                assert "onload=" not in answer
    
    def test_command_injection_prevention(self, client):
        """测试命令注入防护"""
        # 尝试命令注入攻击
        command_payloads = [
            "; ls -la",
            "| cat /etc/passwd",
            "&& rm -rf /",
            "`whoami`",
            "$(cat /etc/hosts)",
            "; curl http://malicious.com"
        ]
        
        for payload in command_payloads:
            response = client.post(
                "/api/qa/ask",
                json={"question": f"正常问题{payload}"}
            )
            
            # 系统应该正常处理，不执行系统命令
            assert response.status_code in [200, 422, 500]
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "")
                
                # 答案不应该包含系统命令执行结果
                assert "root:" not in answer  # /etc/passwd内容
                assert "localhost" not in answer or "127.0.0.1" not in answer  # /etc/hosts内容
    
    def test_path_traversal_prevention(self, client):
        """测试路径遍历攻击防护"""
        # 尝试路径遍历攻击
        path_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "../../../../root/.ssh/id_rsa",
            "..%2F..%2F..%2Fetc%2Fpasswd"  # URL编码
        ]
        
        for payload in path_payloads:
            # 测试文档处理接口
            response = client.post(
                "/api/documents/process",
                json={"file_path": payload}
            )
            
            # 应该返回文件不存在或权限错误，而不是系统文件内容
            assert response.status_code in [404, 403, 422]
            
            if response.status_code != 422:  # 如果不是验证错误
                data = response.json()
                assert "root:" not in data.get("detail", "")
    
    def test_large_input_handling(self, client):
        """测试大输入处理"""
        # 测试超长问题
        very_long_question = "A" * 10000  # 10KB的问题
        
        response = client.post(
            "/api/qa/ask",
            json={"question": very_long_question}
        )
        
        # 系统应该拒绝或截断过长的输入
        assert response.status_code in [422, 413, 400]  # 验证错误或请求过大
    
    def test_malformed_json_handling(self, client):
        """测试畸形JSON处理"""
        malformed_payloads = [
            '{"question": "test"',  # 不完整的JSON
            '{"question": "test", "extra": }',  # 语法错误
            '{"question": null}',  # null值
            '{"question": 123}',  # 错误类型
        ]
        
        for payload in malformed_payloads:
            response = client.post(
                "/api/qa/ask",
                data=payload,
                headers={"Content-Type": "application/json"}
            )
            
            # 应该返回适当的错误状态码
            assert response.status_code in [422, 400]


class TestAuthenticationSecurity:
    """认证安全测试"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_api_key_protection(self, client):
        """测试API密钥保护"""
        # 如果系统配置了API密钥，测试保护机制
        with patch('src.config.settings.get_settings') as mock_settings:
            mock_settings.return_value.api_key = "test-api-key"
            
            # 不提供API密钥的请求
            response = client.post(
                "/api/qa/ask",
                json={"question": "测试问题"}
            )
            
            # 根据中间件配置，可能返回401或正常处理
            if response.status_code == 401:
                data = response.json()
                assert "未授权" in data.get("message", "") or "unauthorized" in data.get("message", "").lower()
    
    def test_rate_limiting(self, client):
        """测试速率限制"""
        # 快速发送多个请求测试速率限制
        responses = []
        for i in range(20):  # 发送20个快速请求
            response = client.post(
                "/api/qa/ask",
                json={"question": f"速率测试问题{i}"}
            )
            responses.append(response.status_code)
        
        # 检查是否有速率限制响应
        rate_limited = any(status == 429 for status in responses)
        
        # 注意：如果没有配置速率限制，这个测试可能会通过
        # 这里主要是验证系统不会因为快速请求而崩溃
        assert all(status in [200, 429, 422, 500] for status in responses)
    
    def test_cors_headers(self, client):
        """测试CORS头部安全"""
        response = client.options("/api/qa/ask")
        
        # 检查CORS头部
        cors_headers = response.headers
        
        # 验证CORS配置不会过于宽松
        if "access-control-allow-origin" in cors_headers:
            origin = cors_headers["access-control-allow-origin"]
            # 如果不是通配符，应该是具体的域名
            if origin != "*":
                assert origin.startswith("http://") or origin.startswith("https://")


class TestDataProtectionSecurity:
    """数据保护安全测试"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_sensitive_data_exposure(self, client):
        """测试敏感数据暴露"""
        # 测试系统信息接口不暴露敏感信息
        response = client.get("/api/system/stats")
        
        if response.status_code == 200:
            data = response.json()
            
            # 检查不应该暴露的敏感信息
            sensitive_keys = [
                "password", "secret", "key", "token", 
                "private", "credential", "auth"
            ]
            
            def check_sensitive_data(obj, path=""):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        current_path = f"{path}.{key}" if path else key
                        
                        # 检查键名是否包含敏感词
                        for sensitive in sensitive_keys:
                            if sensitive in key.lower():
                                # 如果是敏感键，值应该被隐藏或脱敏
                                if isinstance(value, str) and len(value) > 0:
                                    assert "*" in value or "***" in value or value == "[HIDDEN]"
                        
                        check_sensitive_data(value, current_path)
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        check_sensitive_data(item, f"{path}[{i}]")
            
            check_sensitive_data(data)
    
    def test_error_message_information_disclosure(self, client):
        """测试错误消息信息泄露"""
        # 尝试触发各种错误，检查错误消息是否泄露敏感信息
        error_triggers = [
            {"endpoint": "/api/documents/process", "data": {"file_path": "/nonexistent/file.txt"}},
            {"endpoint": "/api/qa/ask", "data": {"question": ""}},
            {"endpoint": "/api/system/config", "method": "PUT", "data": {"invalid": "data"}},
        ]
        
        for trigger in error_triggers:
            method = trigger.get("method", "POST").lower()
            
            if method == "post":
                response = client.post(trigger["endpoint"], json=trigger["data"])
            elif method == "put":
                response = client.put(trigger["endpoint"], json=trigger["data"])
            else:
                response = client.get(trigger["endpoint"])
            
            if response.status_code >= 400:
                data = response.json()
                error_message = data.get("detail", "") or data.get("message", "")
                
                # 错误消息不应该包含敏感信息
                sensitive_patterns = [
                    "/home/", "/root/", "C:\\",  # 文件路径
                    "password", "secret", "key",  # 敏感词
                    "Traceback", "File \"",  # Python堆栈跟踪
                    "at line", "in function",  # 代码位置信息
                ]
                
                for pattern in sensitive_patterns:
                    assert pattern not in error_message
    
    def test_file_upload_security(self, client):
        """测试文件上传安全"""
        # 测试恶意文件上传
        malicious_files = [
            {"name": "malicious.exe", "content": b"MZ\x90\x00", "type": "application/octet-stream"},
            {"name": "script.js", "content": b"alert('XSS')", "type": "application/javascript"},
            {"name": "shell.php", "content": b"<?php system($_GET['cmd']); ?>", "type": "application/x-php"},
            {"name": "../../../etc/passwd", "content": b"root:x:0:0:root:/root:/bin/bash", "type": "text/plain"},
        ]
        
        for file_info in malicious_files:
            file_obj = io.BytesIO(file_info["content"])
            
            response = client.post(
                "/api/documents/upload",
                files={"file": (file_info["name"], file_obj, file_info["type"])}
            )
            
            # 系统应该拒绝恶意文件或不支持的格式
            if response.status_code == 200:
                # 如果上传成功，文件名应该被清理
                data = response.json()
                uploaded_name = data.get("filename", "")
                
                # 不应该包含路径遍历字符
                assert "../" not in uploaded_name
                assert "..\\" not in uploaded_name
            else:
                # 应该返回适当的错误状态码
                assert response.status_code in [400, 415, 422]


class TestSystemHardeningSecurity:
    """系统加固安全测试"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_security_headers(self, client):
        """测试安全头部"""
        response = client.get("/api/health")
        
        headers = response.headers
        
        # 检查重要的安全头部
        security_headers = {
            "x-content-type-options": "nosniff",
            "x-frame-options": "SAMEORIGIN",
            "x-xss-protection": "1; mode=block",
        }
        
        for header, expected_value in security_headers.items():
            if header in headers:
                assert expected_value.lower() in headers[header].lower()
    
    def test_server_information_disclosure(self, client):
        """测试服务器信息泄露"""
        response = client.get("/api/health")
        
        headers = response.headers
        
        # 检查是否泄露服务器信息
        sensitive_headers = ["server", "x-powered-by", "x-aspnet-version"]
        
        for header in sensitive_headers:
            if header in headers:
                # 如果存在这些头部，值不应该包含详细版本信息
                value = headers[header].lower()
                assert "nginx/" not in value or "nginx" == value
                assert "apache/" not in value or "apache" == value
                assert "python/" not in value
    
    def test_directory_traversal_protection(self, client):
        """测试目录遍历保护"""
        # 尝试访问系统文件
        system_paths = [
            "/etc/passwd",
            "/etc/shadow",
            "/proc/version",
            "/windows/system32/config/sam",
            "C:\\boot.ini"
        ]
        
        for path in system_paths:
            # 尝试通过不同接口访问系统文件
            response = client.get(f"/static/{path}")
            
            # 应该返回404或403，而不是文件内容
            assert response.status_code in [404, 403]
            
            # 响应不应该包含系统文件内容
            if response.status_code != 404:
                content = response.text
                assert "root:" not in content
                assert "[boot loader]" not in content
    
    def test_http_method_security(self, client):
        """测试HTTP方法安全"""
        # 测试不应该支持的HTTP方法
        dangerous_methods = ["TRACE", "TRACK", "DEBUG"]
        
        for method in dangerous_methods:
            response = client.request(method, "/api/health")
            
            # 应该返回405 Method Not Allowed
            assert response.status_code == 405


class TestBusinessLogicSecurity:
    """业务逻辑安全测试"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_resource_exhaustion_protection(self, client):
        """测试资源耗尽保护"""
        # 测试大量并发请求
        import threading
        import time
        
        results = []
        
        def make_request():
            try:
                response = client.post(
                    "/api/qa/ask",
                    json={"question": "资源测试问题"},
                    timeout=5
                )
                results.append(response.status_code)
            except Exception as e:
                results.append(0)  # 连接失败
        
        # 创建多个线程同时发送请求
        threads = []
        for _ in range(50):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # 等待所有请求完成
        for thread in threads:
            thread.join(timeout=10)
        
        # 系统应该能够处理并发请求而不崩溃
        successful_requests = sum(1 for status in results if status == 200)
        failed_requests = sum(1 for status in results if status == 0)
        
        # 至少应该有一些成功的请求
        assert successful_requests > 0
        # 失败请求不应该超过总数的50%
        assert failed_requests < len(results) * 0.5
    
    def test_data_validation_bypass(self, client):
        """测试数据验证绕过"""
        # 尝试绕过数据验证
        bypass_attempts = [
            {"question": None},  # null值
            {"question": []},    # 错误类型
            {"question": {"nested": "object"}},  # 嵌套对象
            {"k": -1},          # 负数
            {"k": 999999},      # 超大数值
            {"similarity_threshold": -1.0},  # 超出范围
            {"similarity_threshold": 2.0},   # 超出范围
        ]
        
        for payload in bypass_attempts:
            response = client.post("/api/qa/ask", json=payload)
            
            # 应该返回验证错误
            assert response.status_code == 422
    
    def test_privilege_escalation_prevention(self, client):
        """测试权限提升防护"""
        # 尝试访问管理员功能
        admin_endpoints = [
            "/api/system/shutdown",
            "/api/system/config",
            "/api/system/cache",
        ]
        
        for endpoint in admin_endpoints:
            # 不提供认证信息尝试访问
            response = client.post(endpoint)
            
            # 应该要求认证或返回权限错误
            assert response.status_code in [401, 403, 405, 422]


class TestSecurityConfiguration:
    """安全配置测试"""
    
    def test_default_security_settings(self):
        """测试默认安全设置"""
        from src.config.settings import get_settings
        
        settings = get_settings()
        
        # 检查安全相关的默认配置
        assert settings.debug is False or os.getenv("DEBUG", "").lower() == "true"
        
        # 如果设置了API密钥，应该不为空
        if hasattr(settings, 'api_key') and settings.api_key:
            assert len(settings.api_key) >= 8  # 最小长度要求
    
    def test_environment_variable_security(self):
        """测试环境变量安全"""
        import os
        
        # 检查敏感环境变量是否存在
        sensitive_vars = [
            "API_KEY", "SECRET_KEY", "DATABASE_PASSWORD",
            "REDIS_PASSWORD", "JWT_SECRET"
        ]
        
        for var in sensitive_vars:
            value = os.getenv(var)
            if value:
                # 敏感变量不应该是默认值或过于简单
                assert value not in ["password", "123456", "admin", "secret"]
                assert len(value) >= 8


# 安全测试辅助函数
def is_safe_response(response_text: str) -> bool:
    """检查响应是否安全"""
    dangerous_patterns = [
        "<script>", "javascript:", "onerror=", "onload=",
        "eval(", "document.cookie", "window.location",
        "/etc/passwd", "root:", "admin:",
        "DROP TABLE", "DELETE FROM", "INSERT INTO"
    ]
    
    text_lower = response_text.lower()
    return not any(pattern.lower() in text_lower for pattern in dangerous_patterns)


def sanitize_input(user_input: str) -> str:
    """输入清理示例函数"""
    import html
    import re
    
    # HTML转义
    sanitized = html.escape(user_input)
    
    # 移除潜在的脚本标签
    sanitized = re.sub(r'<script.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
    
    # 移除JavaScript协议
    sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
    
    return sanitized