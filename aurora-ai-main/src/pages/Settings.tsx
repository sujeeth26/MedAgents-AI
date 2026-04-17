import { useState } from "react";
import { User, Bell, Settings as SettingsIcon } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const Settings = () => {
  const [emailNotifications, setEmailNotifications] = useState(true);
  const [pushNotifications, setPushNotifications] = useState(false);
  const [twoFactor, setTwoFactor] = useState(false);

  return (
    <div className="min-h-screen p-8 animate-fade-in">
      <div className="max-w-4xl mx-auto space-y-8">
        {/* Header */}
        <div>
          <h1 className="text-4xl font-bold mb-2 gradient-text">Settings</h1>
          <p className="text-muted-foreground">Manage your account and preferences</p>
        </div>

        {/* Profile Section */}
        <Card className="glass-card border-border/50">
          <CardHeader>
            <div className="flex items-center gap-3">
              <User className="h-5 w-5 text-primary" />
              <div>
                <CardTitle>Profile Settings</CardTitle>
                <CardDescription>Update your personal information</CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="firstName">First Name</Label>
                <Input id="firstName" placeholder="John" className="glass-card" />
              </div>
              <div className="space-y-2">
                <Label htmlFor="lastName">Last Name</Label>
                <Input id="lastName" placeholder="Doe" className="glass-card" />
              </div>
            </div>
            <div className="space-y-2">
              <Label htmlFor="email">Email</Label>
              <Input
                id="email"
                type="email"
                placeholder="john.doe@example.com"
                className="glass-card"
              />
            </div>
            <Button className="bg-primary hover:bg-primary/90">Save Changes</Button>
          </CardContent>
        </Card>

        {/* Notifications */}
        <Card className="glass-card border-border/50">
          <CardHeader>
            <div className="flex items-center gap-3">
              <Bell className="h-5 w-5 text-primary" />
              <div>
                <CardTitle>Notifications</CardTitle>
                <CardDescription>Configure how you receive updates</CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label>Email Notifications</Label>
                <p className="text-sm text-muted-foreground">
                  Receive email updates about your activity
                </p>
              </div>
              <Switch checked={emailNotifications} onCheckedChange={setEmailNotifications} />
            </div>
            <Separator />
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label>Push Notifications</Label>
                <p className="text-sm text-muted-foreground">
                  Get push notifications for important updates
                </p>
              </div>
              <Switch checked={pushNotifications} onCheckedChange={setPushNotifications} />
            </div>
          </CardContent>
        </Card>

        {/* Preferences Tabs */}
        <Card className="glass-card border-border/50">
          <CardHeader>
            <div className="flex items-center gap-3">
              <SettingsIcon className="h-5 w-5 text-primary" />
              <div>
                <CardTitle>Preferences</CardTitle>
                <CardDescription>Customize your experience</CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="security" className="w-full">
              <TabsList className="grid w-full grid-cols-4">
                <TabsTrigger value="security">Security</TabsTrigger>
                <TabsTrigger value="appearance">Appearance</TabsTrigger>
                <TabsTrigger value="shortcuts">Shortcuts</TabsTrigger>
                <TabsTrigger value="features">Features</TabsTrigger>
              </TabsList>

              {/* Security Tab */}
              <TabsContent value="security" className="space-y-6 mt-6">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Two-Factor Authentication</Label>
                    <p className="text-sm text-muted-foreground">
                      Add an extra layer of security to your account
                    </p>
                  </div>
                  <Switch checked={twoFactor} onCheckedChange={setTwoFactor} />
                </div>
                <Separator />
                <div className="space-y-2">
                  <Label>Change Password</Label>
                  <div className="space-y-2">
                    <Input type="password" placeholder="Current password" className="glass-card" />
                    <Input type="password" placeholder="New password" className="glass-card" />
                    <Input
                      type="password"
                      placeholder="Confirm new password"
                      className="glass-card"
                    />
                  </div>
                  <Button variant="outline" className="mt-2">
                    Update Password
                  </Button>
                </div>
              </TabsContent>

              {/* Appearance Tab */}
              <TabsContent value="appearance" className="space-y-6 mt-6">
                <div className="space-y-2">
                  <Label>Theme</Label>
                  <div className="grid grid-cols-3 gap-3">
                    <Button variant="outline" className="glass-card">
                      Light
                    </Button>
                    <Button variant="outline" className="glass-card bg-primary/10 border-primary/20">
                      Dark
                    </Button>
                    <Button variant="outline" className="glass-card">
                      System
                    </Button>
                  </div>
                </div>
                <Separator />
                <div className="space-y-2">
                  <Label>Interface Density</Label>
                  <div className="grid grid-cols-2 gap-3">
                    <Button variant="outline" className="glass-card">
                      Compact
                    </Button>
                    <Button variant="outline" className="glass-card bg-primary/10 border-primary/20">
                      Comfortable
                    </Button>
                  </div>
                </div>
              </TabsContent>

              {/* Shortcuts Tab */}
              <TabsContent value="shortcuts" className="space-y-3 mt-6">
                {[
                  { keys: ["⌘", "N"], action: "New chat" },
                  { keys: ["⌘", "K"], action: "Open command palette" },
                  { keys: ["⌘", "Enter"], action: "Send message" },
                  { keys: ["/"], action: "Slash commands" },
                  { keys: ["⌘", "B"], action: "Toggle theme" },
                  { keys: ["⌘", "F"], action: "Search conversations" },
                ].map((shortcut, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between p-3 rounded-lg hover:bg-muted/50 transition-colors"
                  >
                    <span className="text-sm text-muted-foreground">{shortcut.action}</span>
                    <div className="flex gap-1">
                      {shortcut.keys.map((key, j) => (
                        <kbd
                          key={j}
                          className="px-2 py-1 text-xs bg-muted border border-border rounded"
                        >
                          {key}
                        </kbd>
                      ))}
                    </div>
                  </div>
                ))}
              </TabsContent>

              {/* Features Tab */}
              <TabsContent value="features" className="space-y-6 mt-6">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Auto-save Conversations</Label>
                    <p className="text-sm text-muted-foreground">
                      Automatically save your conversations
                    </p>
                  </div>
                  <Switch defaultChecked />
                </div>
                <Separator />
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Sound Effects</Label>
                    <p className="text-sm text-muted-foreground">
                      Play sounds for message notifications
                    </p>
                  </div>
                  <Switch />
                </div>
                <Separator />
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Show Timestamps</Label>
                    <p className="text-sm text-muted-foreground">
                      Display timestamps on messages
                    </p>
                  </div>
                  <Switch defaultChecked />
                </div>
                <Separator />
                <div className="space-y-2">
                  <Label>Data Management</Label>
                  <div className="flex gap-2">
                    <Button variant="outline" size="sm">
                      Export Data
                    </Button>
                    <Button variant="outline" size="sm">
                      Clear History
                    </Button>
                    <Button variant="destructive" size="sm">
                      Delete Account
                    </Button>
                  </div>
                </div>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Settings;
