"use client";

import { useState } from "react";
import {
  Key,
  Plus,
  Copy,
  Trash2,
  Eye,
  EyeOff,
  Shield,
  Server,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";

interface ApiKeyEntry {
  id: string;
  name: string;
  key: string;
  created: string;
  lastUsed: string | null;
}

const initialKeys: ApiKeyEntry[] = [
  {
    id: "key_001",
    name: "Production App",
    key: "vv_prod_sk_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
    created: "2026-01-15",
    lastUsed: "2026-02-07",
  },
  {
    id: "key_002",
    name: "Development",
    key: "vv_dev_sk_q7r8s9t0u1v2w3x4y5z6a7b8c9d0e1f2",
    created: "2026-02-01",
    lastUsed: "2026-02-06",
  },
];

export default function SettingsPage() {
  const [apiKeys, setApiKeys] = useState<ApiKeyEntry[]>(initialKeys);
  const [showKeys, setShowKeys] = useState<Record<string, boolean>>({});
  const [newKeyName, setNewKeyName] = useState("");
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [apiUrl, setApiUrl] = useState("http://localhost:8000");
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [wsEnabled, setWsEnabled] = useState(true);

  const toggleKeyVisibility = (id: string) => {
    setShowKeys((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const maskKey = (key: string) => {
    return key.substring(0, 10) + "..." + key.substring(key.length - 4);
  };

  const handleCreateKey = () => {
    if (!newKeyName.trim()) return;
    const newKey: ApiKeyEntry = {
      id: `key_${Date.now()}`,
      name: newKeyName,
      key: `vv_new_sk_${Array.from({ length: 32 }, () =>
        "abcdefghijklmnopqrstuvwxyz0123456789"[Math.floor(Math.random() * 36)]
      ).join("")}`,
      created: new Date().toISOString().split("T")[0],
      lastUsed: null,
    };
    setApiKeys((prev) => [...prev, newKey]);
    setNewKeyName("");
    setShowCreateForm(false);
    setShowKeys((prev) => ({ ...prev, [newKey.id]: true }));
  };

  const handleDeleteKey = (id: string) => {
    setApiKeys((prev) => prev.filter((k) => k.id !== id));
  };

  const handleCopyKey = (key: string) => {
    navigator.clipboard.writeText(key);
  };

  return (
    <div className="p-6 lg:p-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold tracking-tight">Settings</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Manage API keys and application preferences
        </p>
      </div>

      <div className="mx-auto max-w-3xl space-y-6">
        {/* API Keys */}
        <Card className="border-border/50 bg-card/50">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Key className="h-4 w-4 text-primary" />
                <CardTitle className="text-base">API Keys</CardTitle>
              </div>
              <Button
                size="sm"
                className="gap-1.5"
                onClick={() => setShowCreateForm(true)}
              >
                <Plus className="h-3.5 w-3.5" />
                New Key
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Create Form */}
            {showCreateForm && (
              <div className="flex items-end gap-2 rounded-md border border-primary/20 bg-primary/5 p-3">
                <div className="flex-1">
                  <Label className="text-xs text-muted-foreground">
                    Key Name
                  </Label>
                  <Input
                    placeholder="e.g. Production, Staging..."
                    value={newKeyName}
                    onChange={(e) => setNewKeyName(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && handleCreateKey()}
                    className="mt-1 bg-background/50"
                  />
                </div>
                <Button size="sm" onClick={handleCreateKey}>
                  Create
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => setShowCreateForm(false)}
                >
                  Cancel
                </Button>
              </div>
            )}

            {/* Key List */}
            {apiKeys.map((apiKey) => (
              <div
                key={apiKey.id}
                className="flex items-center gap-3 rounded-md border border-border/50 bg-background/30 p-3"
              >
                <Shield className="h-4 w-4 shrink-0 text-muted-foreground" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium">{apiKey.name}</p>
                  <p className="mt-0.5 font-mono text-xs text-muted-foreground">
                    {showKeys[apiKey.id] ? apiKey.key : maskKey(apiKey.key)}
                  </p>
                  <p className="mt-1 text-[10px] text-muted-foreground/60">
                    Created {apiKey.created}
                    {apiKey.lastUsed && ` · Last used ${apiKey.lastUsed}`}
                  </p>
                </div>
                <div className="flex items-center gap-1">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7"
                    onClick={() => toggleKeyVisibility(apiKey.id)}
                  >
                    {showKeys[apiKey.id] ? (
                      <EyeOff className="h-3.5 w-3.5" />
                    ) : (
                      <Eye className="h-3.5 w-3.5" />
                    )}
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7"
                    onClick={() => handleCopyKey(apiKey.key)}
                  >
                    <Copy className="h-3.5 w-3.5" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7 text-muted-foreground hover:text-destructive"
                    onClick={() => handleDeleteKey(apiKey.id)}
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </Button>
                </div>
              </div>
            ))}

            {apiKeys.length === 0 && (
              <p className="py-4 text-center text-sm text-muted-foreground">
                No API keys created yet
              </p>
            )}
          </CardContent>
        </Card>

        {/* Connection Settings */}
        <Card className="border-border/50 bg-card/50">
          <CardHeader>
            <div className="flex items-center gap-2">
              <Server className="h-4 w-4 text-primary" />
              <CardTitle className="text-base">Connection</CardTitle>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label className="text-xs text-muted-foreground">
                Backend API URL
              </Label>
              <Input
                value={apiUrl}
                onChange={(e) => setApiUrl(e.target.value)}
                className="mt-1 font-mono text-sm bg-background/50"
              />
            </div>
            <Separator className="bg-border/50" />
            <div className="flex items-center justify-between">
              <div>
                <Label className="text-sm">Auto-refresh Dashboard</Label>
                <p className="text-xs text-muted-foreground">
                  Automatically refresh stats every 30 seconds
                </p>
              </div>
              <Switch
                checked={autoRefresh}
                onCheckedChange={setAutoRefresh}
              />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <Label className="text-sm">WebSocket Connection</Label>
                <p className="text-xs text-muted-foreground">
                  Real-time processing progress updates
                </p>
              </div>
              <Switch checked={wsEnabled} onCheckedChange={setWsEnabled} />
            </div>
          </CardContent>
        </Card>

        {/* About */}
        <Card className="border-border/50 bg-card/50">
          <CardContent className="p-5">
            <div className="text-center">
              <p className="font-mono text-xs text-muted-foreground">
                VaultVision v0.1.0-mvp
              </p>
              <p className="mt-1 text-[10px] text-muted-foreground/60">
                Vault Sync AI LLC &middot; Baton Rouge, LA
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
