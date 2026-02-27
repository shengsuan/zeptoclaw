//! MCP client with HTTP transport.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;

use super::protocol::*;

/// MCP client for communicating with MCP servers over HTTP.
pub struct McpClient {
    /// Server URL (base endpoint).
    url: String,
    /// HTTP client with timeout.
    http: reqwest::Client,
    /// Atomic request ID counter.
    next_id: AtomicU64,
    /// Cached tool definitions.
    tools_cache: Arc<RwLock<Option<Vec<McpTool>>>>,
    /// Server name for logging and tool prefixing.
    server_name: String,
}

impl McpClient {
    /// Create a new MCP client.
    pub fn new(name: &str, url: &str, timeout_secs: u64) -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(timeout_secs))
            .build()
            .unwrap_or_default();

        Self {
            url: url.to_string(),
            http,
            next_id: AtomicU64::new(1),
            tools_cache: Arc::new(RwLock::new(None)),
            server_name: name.to_string(),
        }
    }

    /// Get the next unique request ID.
    fn next_request_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Get the server name.
    pub fn server_name(&self) -> &str {
        &self.server_name
    }

    /// Get the server URL.
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Send a JSON-RPC request and return the response.
    async fn send_request(&self, request: &McpRequest) -> Result<McpResponse, String> {
        let resp = self
            .http
            .post(&self.url)
            .json(request)
            .send()
            .await
            .map_err(|e| format!("HTTP request failed: {}", e))?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("HTTP {} from MCP server: {}", status, body));
        }

        resp.json::<McpResponse>()
            .await
            .map_err(|e| format!("Failed to parse MCP response: {}", e))
    }

    /// Send the initialize handshake.
    pub async fn initialize(&self) -> Result<serde_json::Value, String> {
        let params = InitializeParams::default();
        let request = McpRequest::new(
            self.next_request_id(),
            "initialize",
            Some(serde_json::to_value(&params).map_err(|e| e.to_string())?),
        );

        let response = self.send_request(&request).await?;
        if let Some(error) = response.error {
            return Err(format!("MCP initialize error: {}", error.message));
        }

        Ok(response.result.unwrap_or(serde_json::Value::Null))
    }

    /// List available tools (cached after first call).
    pub async fn list_tools(&self) -> Result<Vec<McpTool>, String> {
        // Check cache first
        {
            let cache = self.tools_cache.read().await;
            if let Some(ref tools) = *cache {
                return Ok(tools.clone());
            }
        }

        let request = McpRequest::new(self.next_request_id(), "tools/list", None);
        let response = self.send_request(&request).await?;

        if let Some(error) = response.error {
            return Err(format!("MCP tools/list error: {}", error.message));
        }

        let result: ListToolsResult =
            serde_json::from_value(response.result.ok_or("No result in tools/list response")?)
                .map_err(|e| format!("Failed to parse tools list: {}", e))?;

        // Update cache
        let tools = result.tools;
        {
            let mut cache = self.tools_cache.write().await;
            *cache = Some(tools.clone());
        }

        Ok(tools)
    }

    /// Call a tool by name with arguments.
    pub async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<CallToolResult, String> {
        let params = serde_json::json!({
            "name": name,
            "arguments": arguments,
        });

        let request = McpRequest::new(self.next_request_id(), "tools/call", Some(params));
        let response = self.send_request(&request).await?;

        if let Some(error) = response.error {
            return Err(format!("MCP tools/call error: {}", error.message));
        }

        let result: CallToolResult =
            serde_json::from_value(response.result.ok_or("No result in tools/call response")?)
                .map_err(|e| format!("Failed to parse tool call result: {}", e))?;

        Ok(result)
    }

    /// Invalidate the tools cache (force re-fetch on next list_tools).
    pub async fn invalidate_cache(&self) {
        let mut cache = self.tools_cache.write().await;
        *cache = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = McpClient::new("test-server", "http://localhost:8080", 30);
        assert_eq!(client.server_name(), "test-server");
        assert_eq!(client.url(), "http://localhost:8080");
    }

    #[test]
    fn test_request_id_increments() {
        let client = McpClient::new("test", "http://localhost:8080", 30);
        let id1 = client.next_request_id();
        let id2 = client.next_request_id();
        let id3 = client.next_request_id();
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(id3, 3);
    }

    #[tokio::test]
    async fn test_invalidate_cache() {
        let client = McpClient::new("test", "http://localhost:8080", 30);

        // Manually populate cache
        {
            let mut cache = client.tools_cache.write().await;
            *cache = Some(vec![McpTool {
                name: "test_tool".to_string(),
                description: Some("A test tool".to_string()),
                input_schema: serde_json::json!({"type": "object"}),
            }]);
        }

        // Verify cache is populated
        {
            let cache = client.tools_cache.read().await;
            assert!(cache.is_some());
        }

        // Invalidate
        client.invalidate_cache().await;

        // Verify cache is cleared
        {
            let cache = client.tools_cache.read().await;
            assert!(cache.is_none());
        }
    }

    #[test]
    fn test_client_default_timeout() {
        // Verify client can be created with various timeouts without panic
        let _c1 = McpClient::new("fast", "http://localhost:8080", 5);
        let _c2 = McpClient::new("slow", "http://localhost:8080", 120);
        let _c3 = McpClient::new("very-slow", "http://localhost:8080", 600);
    }

    #[test]
    fn test_server_name_accessor() {
        let client = McpClient::new("my-mcp-server", "http://example.com", 30);
        assert_eq!(client.server_name(), "my-mcp-server");
    }

    #[test]
    fn test_url_accessor() {
        let client = McpClient::new("test", "https://mcp.example.com/rpc", 30);
        assert_eq!(client.url(), "https://mcp.example.com/rpc");
    }
}
