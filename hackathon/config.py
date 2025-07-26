# Add these constants at the top of your existing config or create new file
class DashboardConfig:
    # Auto-reply settings
    AUTO_REPLY_THRESHOLD = 0.9  # Confidence threshold for auto-reply
    MAX_AUTO_RESPONSE_TIME = 7200  # 2 hours in seconds
    
    # Dashboard settings
    DEFAULT_TIME_FILTER = "week"  # week/month/all
    TOP_CATEGORIES_LIMIT = 5