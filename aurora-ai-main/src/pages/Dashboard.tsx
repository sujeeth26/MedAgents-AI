import { Activity, MessageSquare, Users, Zap, TrendingUp, Clock } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

const Dashboard = () => {
  const kpis = [
    {
      title: "Total Chats",
      value: "1,234",
      change: "+12.5%",
      icon: MessageSquare,
      color: "text-primary",
    },
    {
      title: "Active Users",
      value: "892",
      change: "+8.2%",
      icon: Users,
      color: "text-accent",
    },
    {
      title: "Requests/Min",
      value: "45.3",
      change: "+15.8%",
      icon: Activity,
      color: "text-primary",
    },
    {
      title: "Avg Latency",
      value: "234ms",
      change: "-5.3%",
      icon: Zap,
      color: "text-accent",
    },
  ];

  const recentActivity = [
    { user: "Alice Chen", action: "Started new chat", time: "2 min ago" },
    { user: "Bob Smith", action: "Completed conversation", time: "5 min ago" },
    { user: "Carol White", action: "Generated report", time: "12 min ago" },
    { user: "David Lee", action: "Updated settings", time: "18 min ago" },
    { user: "Eve Martinez", action: "Started new chat", time: "23 min ago" },
  ];

  return (
    <div className="min-h-screen p-8 animate-fade-in">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div>
          <h1 className="text-4xl font-bold mb-2 gradient-text">Dashboard</h1>
          <p className="text-muted-foreground">Monitor your AI chat performance and analytics</p>
        </div>

        {/* KPI Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {kpis.map((kpi) => (
            <Card key={kpi.title} className="glass-card border-border/50 hover-lift">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">{kpi.title}</CardTitle>
                <kpi.icon className={`h-4 w-4 ${kpi.color}`} />
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold mb-1">{kpi.value}</div>
                <p className="text-xs text-muted-foreground flex items-center gap-1">
                  <TrendingUp className="h-3 w-3" />
                  {kpi.change} from last period
                </p>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Charts Section */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Usage Chart */}
          <Card className="glass-card border-border/50">
            <CardHeader>
              <CardTitle>Usage Overview</CardTitle>
              <CardDescription>Chat activity over the past 7 days</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-64 flex items-end justify-between gap-2">
                {[65, 85, 72, 90, 78, 95, 88].map((height, i) => (
                  <div key={i} className="flex-1 flex flex-col items-center gap-2">
                    <div
                      className="w-full bg-gradient-primary rounded-t-lg transition-all hover:opacity-80"
                      style={{ height: `${height}%` }}
                    />
                    <div className="text-xs text-muted-foreground">
                      {["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][i]}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Performance Metrics */}
          <Card className="glass-card border-border/50">
            <CardHeader>
              <CardTitle>Performance Metrics</CardTitle>
              <CardDescription>Response times and throughput</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">Response Time</span>
                    <span className="text-sm text-muted-foreground">234ms avg</span>
                  </div>
                  <div className="h-3 bg-muted rounded-full overflow-hidden">
                    <div className="h-full bg-gradient-primary w-[75%] rounded-full" />
                  </div>
                </div>
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">Success Rate</span>
                    <span className="text-sm text-muted-foreground">99.2%</span>
                  </div>
                  <div className="h-3 bg-muted rounded-full overflow-hidden">
                    <div className="h-full bg-gradient-primary w-[99%] rounded-full" />
                  </div>
                </div>
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">Token Usage</span>
                    <span className="text-sm text-muted-foreground">2.3M tokens</span>
                  </div>
                  <div className="h-3 bg-muted rounded-full overflow-hidden">
                    <div className="h-full bg-gradient-primary w-[65%] rounded-full" />
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Recent Activity */}
        <Card className="glass-card border-border/50">
          <CardHeader>
            <CardTitle>Recent Activity</CardTitle>
            <CardDescription>Latest user interactions and system events</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {recentActivity.map((activity, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between p-3 rounded-lg hover:bg-muted/50 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <div className="h-10 w-10 rounded-full bg-gradient-primary flex items-center justify-center text-primary-foreground font-semibold text-sm">
                      {activity.user.split(" ").map((n) => n[0]).join("")}
                    </div>
                    <div>
                      <div className="font-medium text-sm">{activity.user}</div>
                      <div className="text-xs text-muted-foreground">{activity.action}</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-1 text-xs text-muted-foreground">
                    <Clock className="h-3 w-3" />
                    {activity.time}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Dashboard;
