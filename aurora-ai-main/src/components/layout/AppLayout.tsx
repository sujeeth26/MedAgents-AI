import { Link, useLocation, useNavigate } from "react-router-dom";
import { MessageSquare, LayoutDashboard, Settings, Sparkles, Menu, Plus, Zap, TrendingUp, Activity } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useState, useEffect } from "react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";

interface AppLayoutProps {
  children: React.ReactNode;
}

const AppLayout = ({ children }: AppLayoutProps) => {
  const location = useLocation();
  const navigate = useNavigate();
  const [collapsed, setCollapsed] = useState(false);
  const [activeStats, setActiveStats] = useState({ chats: 0, agents: 6, uptime: "99.9%" });

  useEffect(() => {
    // Simulate dynamic stats
    const interval = setInterval(() => {
      setActiveStats(prev => ({
        ...prev,
        chats: Math.floor(Math.random() * 50) + 10,
      }));
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const navItems = [
    {
      path: "/",
      icon: MessageSquare,
      label: "Chat",
      badge: activeStats.chats,
      description: "AI Medical Assistant"
    },
    {
      path: "/dashboard",
      icon: LayoutDashboard,
      label: "Dashboard",
      description: "Analytics & Insights"
    },
    {
      path: "/settings",
      icon: Settings,
      label: "Settings",
      description: "Preferences"
    },
  ];

  const isActive = (path: string) => {
    return location.pathname === path;
  };

  return (
    <div className="flex h-screen w-full overflow-hidden bg-slate-50">
      {/* Premium Navigation Sidebar - Light Theme */}
      <div
        className={cn(
          "border-r border-slate-200 bg-white/80 backdrop-blur-xl flex flex-col transition-all duration-500 shadow-xl z-30",
          collapsed ? "w-20" : "w-72"
        )}
      >
        {/* Logo Section with Animation */}
        <div className="h-20 border-b border-slate-100 flex items-center px-4 relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-r from-blue-50/50 via-indigo-50/50 to-blue-50/50 animate-gradient-x" />
          <Link to="/" className="flex items-center gap-3 relative z-10 w-full group">
            <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-blue-600 via-indigo-600 to-violet-600 flex items-center justify-center shadow-lg shadow-blue-500/20 group-hover:scale-110 transition-transform duration-300">
              <Sparkles className="h-6 w-6 text-white" />
            </div>
            {!collapsed && (
              <div className="flex-1">
                <span className="font-bold text-xl bg-gradient-to-r from-blue-600 via-indigo-600 to-blue-600 bg-clip-text text-transparent">
                  MedAgentica
                </span>
                <div className="text-xs text-slate-500">AI Medical Assistant</div>
              </div>
            )}
          </Link>
        </div>

        {/* Toggle Button - Sidebar Collapse/Expand */}
        <div className="p-3 border-b border-slate-100">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setCollapsed(!collapsed)}
            className="w-full hover:bg-slate-100 text-slate-500 hover:text-blue-600 transition-all duration-300"
            title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            <Menu className="h-5 w-5" />
          </Button>
        </div>

        {/* Navigation Items with Premium Styling */}
        <nav className="flex-1 p-3 space-y-2 overflow-y-auto">
          {navItems.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              className={cn(
                "group relative flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-300 overflow-hidden",
                isActive(item.path)
                  ? "bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-100 text-blue-700 shadow-sm"
                  : "hover:bg-slate-50 border border-transparent text-slate-500 hover:text-slate-900"
              )}
            >
              <item.icon className={cn(
                "h-5 w-5 shrink-0 relative z-10 transition-transform duration-300",
                isActive(item.path) ? "text-blue-600" : "text-slate-400 group-hover:text-blue-500 group-hover:scale-110"
              )} />

              {!collapsed && (
                <div className="flex-1 min-w-0 relative z-10">
                  <div className="flex items-center justify-between">
                    <span className={cn(
                      "text-sm font-medium",
                      isActive(item.path) ? "text-blue-900" : "text-slate-600"
                    )}>
                      {item.label}
                    </span>
                    {item.badge !== undefined && (
                      <Badge className="bg-blue-100 text-blue-600 border-blue-200 text-xs px-2 py-0 shadow-none hover:bg-blue-100">
                        {item.badge}
                      </Badge>
                    )}
                  </div>
                  {item.description && (
                    <div className="text-xs text-slate-400 mt-0.5">{item.description}</div>
                  )}
                </div>
              )}
            </Link>
          ))}
        </nav>

        {/* System Stats */}
        {!collapsed && (
          <div className="p-4 border-t border-slate-100 space-y-2 bg-slate-50/50">
            <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
              System Status
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs">
                <div className="flex items-center gap-2 text-slate-500">
                  <Activity className="h-3 w-3 text-green-500" />
                  <span>Uptime</span>
                </div>
                <span className="text-green-600 font-semibold">{activeStats.uptime}</span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <div className="flex items-center gap-2 text-slate-500">
                  <Zap className="h-3 w-3 text-yellow-500" />
                  <span>Agents</span>
                </div>
                <span className="text-yellow-600 font-semibold">{activeStats.agents}</span>
              </div>
            </div>
          </div>
        )}

        {/* User Section with Premium Design */}
        <div className="p-4 border-t border-slate-100">
          <div
            className={cn(
              "flex items-center gap-3 p-3 rounded-xl bg-white border border-slate-200 shadow-sm transition-all duration-300 hover:border-blue-200 hover:shadow-md",
              collapsed && "justify-center"
            )}
          >
            <div className="h-10 w-10 rounded-full bg-gradient-to-br from-blue-500 via-indigo-500 to-violet-500 flex items-center justify-center text-white font-bold text-sm shrink-0 shadow-md shadow-blue-500/20 relative">
              <span>User</span>
              <div className="absolute -top-1 -right-1 h-3 w-3 bg-green-500 rounded-full border-2 border-white" />
            </div>
            {!collapsed && (
              <div className="flex-1 min-w-0">
                <div className="text-sm font-semibold text-slate-900 truncate">Dr. User</div>
                <div className="text-xs text-slate-500 truncate">Licensed Clinician</div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-auto relative bg-slate-50">
        {children}
      </div>
    </div>
  );
};

export default AppLayout;
