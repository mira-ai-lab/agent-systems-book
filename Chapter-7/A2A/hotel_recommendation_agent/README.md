# Hotel Recommendation Agent

按城市与偏好推荐**一个**酒店/住宿（当前为 stub 实现，可替换为真实“百度地图”POI/酒店接口）。

## Run

```bash
python -m agents.hotel_recommendation_agent.server --host 0.0.0.0 --port 9012
```

`.env` 需要配置：`DEPLOYMENT_NAME`、`CHAT_API_KEY`、`CHAT_ENDPOINT`。

