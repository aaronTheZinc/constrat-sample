"use client";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  TrendingUp,
  TrendingDown,
  DollarSign,
  AlertTriangle,
  Upload,
  FileText,
  BarChart3,
} from "lucide-react";

export function DashboardOverview() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-foreground">
            Construction Cost Dashboard
          </h1>
          <p className="text-muted-foreground">
            Optimize your procurement timing and predict construction costs with
            AI-powered insights
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" className="gap-2 bg-transparent">
            <Upload className="h-4 w-4" />
            Import Purchase Order
          </Button>
          <Button className="gap-2">
            <FileText className="h-4 w-4" />
            New Project
          </Button>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Total Project Value
            </CardTitle>
            <DollarSign className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">$2,847,500</div>
            <p className="text-xs text-muted-foreground">
              <span className="inline-flex items-center text-zinc-600">
                <TrendingUp className="h-3 w-3 mr-1" />
                +12.5%
              </span>{" "}
              from last month
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center bg- justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Predicted Savings
            </CardTitle>
            <TrendingDown className="h-4 w-4 text-zinc-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-zinc-600">$184,200</div>
            <p className="text-xs text-muted-foreground">
              Through optimal timing
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Active Projects
            </CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">12</div>
            <p className="text-xs text-muted-foreground">
              8 on schedule, 4 delayed
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Weather Impact
            </CardTitle>
            <AlertTriangle className="h-4 w-4 text-amber-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">Medium</div>
            <p className="text-xs text-muted-foreground">
              Rain expected next week
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Quick Actions & Alerts */}
      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Timing Recommendations</CardTitle>
            <CardDescription>
              AI-powered suggestions for optimal purchase timing
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between p-3 bg-zinc-50 rounded-lg border border-zinc-200">
              <div className="flex items-center gap-3">
                <div className="w-2 h-2 bg-zinc-500 rounded-full"></div>
                <div>
                  <p className="font-medium text-sm">Steel Rebar - Grade 60</p>
                  <p className="text-xs text-muted-foreground">
                    Buy in 2 weeks
                  </p>
                </div>
              </div>
              <Badge variant="secondary" className="bg-zinc-100 text-zinc-700">
                Save 8%
              </Badge>
            </div>

            <div className="flex items-center justify-between p-3 bg-amber-50 rounded-lg border border-amber-200">
              <div className="flex items-center gap-3">
                <div className="w-2 h-2 bg-amber-500 rounded-full"></div>
                <div>
                  <p className="font-medium text-sm">Concrete Mix - 4000 PSI</p>
                  <p className="text-xs text-muted-foreground">Buy now</p>
                </div>
              </div>
              <Badge
                variant="secondary"
                className="bg-amber-100 text-amber-700"
              >
                Price Rising
              </Badge>
            </div>

            <div className="flex items-center justify-between p-3 bg-blue-50 rounded-lg border border-blue-200">
              <div className="flex items-center gap-3">
                <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                <div>
                  <p className="font-medium text-sm">Lumber - 2x4x8 SPF</p>
                  <p className="text-xs text-muted-foreground">Wait 1 month</p>
                </div>
              </div>
              <Badge variant="secondary" className="bg-blue-100 text-blue-700">
                Save 15%
              </Badge>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Recent Purchase Orders</CardTitle>
            <CardDescription>
              Latest imported and processed orders
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between p-3 border rounded-lg">
              <div className="flex items-center gap-3">
                <FileText className="h-8 w-8 text-muted-foreground" />
                <div>
                  <p className="font-medium text-sm">Downtown Office Complex</p>
                  <p className="text-xs text-muted-foreground">
                    PO-2024-0156 • 45 items
                  </p>
                </div>
              </div>
              <div className="text-right">
                <p className="font-medium text-sm">$485,200</p>
                <Badge variant="outline">Processing</Badge>
              </div>
            </div>

            <div className="flex items-center justify-between p-3 border rounded-lg">
              <div className="flex items-center gap-3">
                <FileText className="h-8 w-8 text-muted-foreground" />
                <div>
                  <p className="font-medium text-sm">Residential Development</p>
                  <p className="text-xs text-muted-foreground">
                    PO-2024-0155 • 32 items
                  </p>
                </div>
              </div>
              <div className="text-right">
                <p className="font-medium text-sm">$298,750</p>
                <Badge className="bg-zinc-100 text-zinc-700">Optimized</Badge>
              </div>
            </div>

            <div className="flex items-center justify-between p-3 border rounded-lg">
              <div className="flex items-center gap-3">
                <FileText className="h-8 w-8 text-muted-foreground" />
                <div>
                  <p className="font-medium text-sm">Highway Bridge Repair</p>
                  <p className="text-xs text-muted-foreground">
                    PO-2024-0154 • 28 items
                  </p>
                </div>
              </div>
              <div className="text-right">
                <p className="font-medium text-sm">$156,900</p>
                <Badge className="bg-zinc-100 text-zinc-700">Complete</Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Market Insights Preview */}
      <Card>
        <CardHeader>
          <CardTitle>Market Insights</CardTitle>
          <CardDescription>
            Current market trends affecting construction material costs
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-3">
            <div className="p-4 border rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <h4 className="font-medium">Steel Prices</h4>
                <TrendingUp className="h-4 w-4 text-red-500" />
              </div>
              <p className="text-2xl font-bold">$847/ton</p>
              <p className="text-xs text-red-600">+5.2% this week</p>
            </div>

            <div className="p-4 border rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <h4 className="font-medium">Lumber Prices</h4>
                <TrendingDown className="h-4 w-4 text-zinc-500" />
              </div>
              <p className="text-2xl font-bold">$425/MBF</p>
              <p className="text-xs text-zinc-600">-3.1% this week</p>
            </div>

            <div className="p-4 border rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <h4 className="font-medium">Concrete Prices</h4>
                <TrendingUp className="h-4 w-4 text-amber-500" />
              </div>
              <p className="text-2xl font-bold">$125/yard³</p>
              <p className="text-xs text-amber-600">+1.8% this week</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
