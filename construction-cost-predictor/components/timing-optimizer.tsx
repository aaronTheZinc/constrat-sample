"use client";

import { useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import {
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  ComposedChart,
} from "recharts";
import {
  Calendar,
  Clock,
  Target,
  Zap,
  AlertTriangle,
  CheckCircle,
  DollarSign,
  Timer,
} from "lucide-react";

// Mock timing data
const timelineData = [
  {
    week: "Week 1",
    steel: { price: 847, optimal: false, savings: 0 },
    lumber: { price: 410, optimal: true, savings: 15 },
    concrete: { price: 125, optimal: false, savings: 0 },
  },
  {
    week: "Week 2",
    steel: { price: 835, optimal: true, savings: 12 },
    lumber: { price: 395, optimal: true, savings: 30 },
    concrete: { price: 128, optimal: false, savings: 0 },
  },
  {
    week: "Week 3",
    steel: { price: 820, optimal: true, savings: 27 },
    lumber: { price: 380, optimal: true, savings: 45 },
    concrete: { price: 122, optimal: true, savings: 8 },
  },
  {
    week: "Week 4",
    steel: { price: 860, optimal: false, savings: 0 },
    lumber: { price: 365, optimal: true, savings: 60 },
    concrete: { price: 135, optimal: false, savings: 0 },
  },
];

const profitForecast = [
  { month: "Jan", baseline: 150000, optimized: 165000, weather: 160000 },
  { month: "Feb", baseline: 180000, optimized: 198000, weather: 185000 },
  { month: "Mar", baseline: 220000, optimized: 245000, weather: 230000 },
  { month: "Apr", baseline: 280000, optimized: 315000, weather: 290000 },
  { month: "May", baseline: 320000, optimized: 365000, weather: 340000 },
  { month: "Jun", baseline: 380000, optimized: 435000, weather: 400000 },
];

const recommendations = [
  {
    material: "Steel Rebar Grade 60",
    currentPrice: 847,
    optimalPrice: 820,
    optimalWeek: "Week 3",
    savings: 27,
    confidence: 85,
    urgency: "medium",
    factors: ["Market volatility", "Supply chain delays"],
  },
  {
    material: "Lumber 2x4x8 SPF",
    currentPrice: 410,
    optimalPrice: 365,
    optimalWeek: "Week 4",
    savings: 45,
    confidence: 78,
    urgency: "low",
    factors: ["Seasonal demand", "Weather patterns"],
  },
  {
    material: "Concrete Mix 4000 PSI",
    currentPrice: 125,
    optimalPrice: 122,
    optimalWeek: "Week 3",
    savings: 3,
    confidence: 82,
    urgency: "high",
    factors: ["Weather forecast", "Production capacity"],
  },
];

const projectTimeline = [
  {
    phase: "Foundation",
    start: "Week 1",
    duration: 2,
    materials: ["concrete", "steel"],
    critical: true,
  },
  {
    phase: "Framing",
    start: "Week 3",
    duration: 3,
    materials: ["lumber", "steel"],
    critical: true,
  },
  {
    phase: "Roofing",
    start: "Week 6",
    duration: 2,
    materials: ["lumber"],
    critical: false,
  },
  {
    phase: "Finishing",
    start: "Week 8",
    duration: 4,
    materials: ["lumber"],
    critical: false,
  },
];

export function TimingOptimizer() {
  const [selectedProject, setSelectedProject] = useState("downtown-office");
  const [optimizationGoal, setOptimizationGoal] = useState("cost");
  const [riskTolerance, setRiskTolerance] = useState([50]);

  const getUrgencyColor = (urgency: string) => {
    switch (urgency) {
      case "high":
        return "bg-red-100 text-red-700";
      case "medium":
        return "bg-amber-100 text-amber-700";
      case "low":
        return "bg-zinc-100 text-zinc-700";
      default:
        return "bg-gray-100 text-gray-700";
    }
  };

  const getUrgencyIcon = (urgency: string) => {
    switch (urgency) {
      case "high":
        return <AlertTriangle className="h-3 w-3" />;
      case "medium":
        return <Clock className="h-3 w-3" />;
      case "low":
        return <CheckCircle className="h-3 w-3" />;
      default:
        return <Timer className="h-3 w-3" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-foreground">
            Timing Optimizer
          </h1>
          <p className="text-muted-foreground">
            AI-powered procurement timing optimization for maximum cost savings
            and profit
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" className="gap-2 bg-transparent">
            <Calendar className="h-4 w-4" />
            Schedule Analysis
          </Button>
          <Button className="gap-2">
            <Target className="h-4 w-4" />
            Optimize Timeline
          </Button>
        </div>
      </div>

      {/* Optimization Controls */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5" />
            Optimization Parameters
          </CardTitle>
          <CardDescription>
            Configure your timing optimization preferences
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-6 md:grid-cols-4">
            <div className="space-y-2">
              <Label htmlFor="project">Project</Label>
              <Select
                value={selectedProject}
                onValueChange={setSelectedProject}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select project" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="downtown-office">
                    Downtown Office Complex
                  </SelectItem>
                  <SelectItem value="residential">
                    Residential Development
                  </SelectItem>
                  <SelectItem value="highway-bridge">
                    Highway Bridge Repair
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="goal">Optimization Goal</Label>
              <Select
                value={optimizationGoal}
                onValueChange={setOptimizationGoal}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select goal" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="cost">Minimize Cost</SelectItem>
                  <SelectItem value="time">Minimize Time</SelectItem>
                  <SelectItem value="profit">Maximize Profit</SelectItem>
                  <SelectItem value="balanced">Balanced Approach</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="budget">Project Budget</Label>
              <Input placeholder="Enter budget" defaultValue="$2,847,500" />
            </div>
            <div className="space-y-2">
              <Label htmlFor="risk">Risk Tolerance: {riskTolerance[0]}%</Label>
              <Slider
                value={riskTolerance}
                onValueChange={setRiskTolerance}
                max={100}
                step={10}
                className="w-full"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      <Tabs defaultValue="timeline" className="space-y-6">
        <TabsList>
          <TabsTrigger value="timeline">Optimal Timeline</TabsTrigger>
          <TabsTrigger value="recommendations">AI Recommendations</TabsTrigger>
          <TabsTrigger value="profit">Profit Forecast</TabsTrigger>
          <TabsTrigger value="scenarios">Scenario Analysis</TabsTrigger>
        </TabsList>

        <TabsContent value="timeline" className="space-y-6">
          {/* Interactive Timeline */}
          <Card>
            <CardHeader>
              <CardTitle>Procurement Timeline Optimization</CardTitle>
              <CardDescription>
                Interactive timeline showing optimal purchase windows for
                maximum savings
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={timelineData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="week" />
                    <YAxis yAxisId="left" />
                    <YAxis yAxisId="right" orientation="right" />
                    <Tooltip
                      content={({ active, payload, label }) => {
                        if (active && payload && payload.length) {
                          return (
                            <div className="bg-white p-3 border rounded-lg shadow-lg">
                              <p className="font-medium">{label}</p>
                              {payload.map((entry, index) => (
                                <div key={index} className="text-sm">
                                  <span style={{ color: entry.color }}>
                                    {entry.name}: ${entry.value}
                                  </span>
                                </div>
                              ))}
                            </div>
                          );
                        }
                        return null;
                      }}
                    />
                    <Bar
                      yAxisId="left"
                      dataKey="steel.price"
                      fill="#ef4444"
                      name="Steel Price"
                    />
                    <Bar
                      yAxisId="left"
                      dataKey="lumber.price"
                      fill="#10b981"
                      name="Lumber Price"
                    />
                    <Line
                      yAxisId="right"
                      type="monotone"
                      dataKey="concrete.price"
                      stroke="#3b82f6"
                      strokeWidth={2}
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-4 grid gap-2 md:grid-cols-3">
                <div className="flex items-center gap-2 text-sm">
                  <div className="w-3 h-3 bg-zinc-500 rounded-full"></div>
                  <span>Optimal Purchase Window</span>
                </div>
                <div className="flex items-center gap-2 text-sm">
                  <div className="w-3 h-3 bg-amber-500 rounded-full"></div>
                  <span>Acceptable Window</span>
                </div>
                <div className="flex items-center gap-2 text-sm">
                  <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                  <span>Avoid Purchase</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Project Phase Timeline */}
          <Card>
            <CardHeader>
              <CardTitle>Project Phase Alignment</CardTitle>
              <CardDescription>
                Align material purchases with project phases for optimal cash
                flow
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {projectTimeline.map((phase, index) => (
                  <div
                    key={index}
                    className="flex items-center gap-4 p-4 border rounded-lg"
                  >
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-medium">{phase.phase}</span>
                        {phase.critical && (
                          <Badge className="bg-red-100 text-red-700">
                            Critical
                          </Badge>
                        )}
                      </div>
                      <div className="text-sm text-muted-foreground">
                        {phase.start} • {phase.duration} weeks • Materials:{" "}
                        {phase.materials.join(", ")}
                      </div>
                    </div>
                    <div className="flex gap-2">
                      {phase.materials.map((material, idx) => (
                        <Badge key={idx} variant="outline">
                          {material}
                        </Badge>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="recommendations" className="space-y-6">
          {/* AI Recommendations */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="h-5 w-5" />
                AI-Powered Timing Recommendations
              </CardTitle>
              <CardDescription>
                Personalized recommendations based on your project requirements
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {recommendations.map((rec, index) => (
                <div key={index} className="p-4 border rounded-lg">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex-1">
                      <h4 className="font-medium">{rec.material}</h4>
                      <div className="flex items-center gap-4 mt-1 text-sm text-muted-foreground">
                        <span>Current: ${rec.currentPrice}</span>
                        <span>Optimal: ${rec.optimalPrice}</span>
                        <span className="text-zinc-600">
                          Save: ${rec.savings}
                        </span>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge className={getUrgencyColor(rec.urgency)}>
                        {getUrgencyIcon(rec.urgency)}
                        <span className="ml-1 capitalize">{rec.urgency}</span>
                      </Badge>
                    </div>
                  </div>
                  <div className="grid gap-3 md:grid-cols-3">
                    <div>
                      <div className="text-sm font-medium">Optimal Timing</div>
                      <div className="text-sm text-muted-foreground">
                        {rec.optimalWeek}
                      </div>
                    </div>
                    <div>
                      <div className="text-sm font-medium">Confidence</div>
                      <div className="text-sm text-muted-foreground">
                        {rec.confidence}%
                      </div>
                    </div>
                    <div>
                      <div className="text-sm font-medium">Key Factors</div>
                      <div className="text-sm text-muted-foreground">
                        {rec.factors.join(", ")}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>

          {/* Quick Actions */}
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Immediate Actions</CardTitle>
                <CardDescription>
                  Actions you should take this week
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center gap-3 p-3 bg-red-50 border border-red-200 rounded-lg">
                  <AlertTriangle className="h-4 w-4 text-red-600" />
                  <div className="flex-1">
                    <div className="font-medium text-red-900">
                      Purchase Concrete Now
                    </div>
                    <div className="text-sm text-red-700">
                      Prices rising 10% next week
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-3 p-3 bg-zinc-50 border border-zinc-200 rounded-lg">
                  <CheckCircle className="h-4 w-4 text-zinc-600" />
                  <div className="flex-1">
                    <div className="font-medium text-zinc-900">
                      Delay Steel Purchase
                    </div>
                    <div className="text-sm text-zinc-700">
                      Wait 2 weeks to save $27/ton
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Optimization Summary</CardTitle>
                <CardDescription>
                  Total impact of timing optimization
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-zinc-600">
                      $184,200
                    </div>
                    <div className="text-sm text-muted-foreground">
                      Total Potential Savings
                    </div>
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Cost Reduction</span>
                      <span className="text-zinc-600">6.5%</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Timeline Impact</span>
                      <span className="text-blue-600">+2 weeks</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Risk Level</span>
                      <span className="text-amber-600">Medium</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="profit" className="space-y-6">
          {/* Profit Forecast */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <DollarSign className="h-5 w-5" />
                Profit Forecast with Timing Optimization
              </CardTitle>
              <CardDescription>
                Compare baseline vs optimized profit projections including
                weather impact
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={profitForecast}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="month" />
                    <YAxis />
                    <Tooltip
                      formatter={(value, name) => [
                        `$${value.toLocaleString()}`,
                        name === "baseline"
                          ? "Baseline"
                          : name === "optimized"
                          ? "Optimized"
                          : "Weather Adjusted",
                      ]}
                    />
                    <Area
                      type="monotone"
                      dataKey="baseline"
                      stackId="1"
                      stroke="#94a3b8"
                      fill="#94a3b8"
                      fillOpacity={0.3}
                    />
                    <Area
                      type="monotone"
                      dataKey="optimized"
                      stackId="2"
                      stroke="#10b981"
                      fill="#10b981"
                      fillOpacity={0.5}
                    />
                    <Line
                      type="monotone"
                      dataKey="weather"
                      stroke="#f59e0b"
                      strokeWidth={2}
                      strokeDasharray="5 5"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-4 grid gap-4 md:grid-cols-3">
                <div className="text-center">
                  <div className="text-lg font-bold">$2,163,000</div>
                  <div className="text-sm text-muted-foreground">
                    Baseline Profit
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-lg font-bold text-zinc-600">
                    $2,523,000
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Optimized Profit
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-lg font-bold text-zinc-600">
                    +$360,000
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Additional Profit
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* ROI Analysis */}
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>ROI Analysis</CardTitle>
                <CardDescription>
                  Return on investment from timing optimization
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex justify-between items-center p-3 border rounded-lg">
                  <span>Implementation Cost</span>
                  <span className="font-medium">$15,000</span>
                </div>
                <div className="flex justify-between items-center p-3 border rounded-lg">
                  <span>Annual Savings</span>
                  <span className="font-medium text-zinc-600">$360,000</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-zinc-50 border border-zinc-200 rounded-lg">
                  <span className="font-medium">ROI</span>
                  <span className="text-lg font-bold text-zinc-600">
                    2,300%
                  </span>
                </div>
                <div className="flex justify-between items-center p-3 border rounded-lg">
                  <span>Payback Period</span>
                  <span className="font-medium">15 days</span>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Risk vs Reward</CardTitle>
                <CardDescription>
                  Balance between potential savings and associated risks
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={[
                        { scenario: "Conservative", savings: 180000, risk: 20 },
                        { scenario: "Balanced", savings: 360000, risk: 50 },
                        { scenario: "Aggressive", savings: 540000, risk: 80 },
                      ]}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="scenario" />
                      <YAxis yAxisId="left" />
                      <YAxis yAxisId="right" orientation="right" />
                      <Tooltip />
                      <Bar
                        yAxisId="left"
                        dataKey="savings"
                        fill="#10b981"
                        name="Savings ($)"
                      />
                      <Line
                        yAxisId="right"
                        type="monotone"
                        dataKey="risk"
                        stroke="#ef4444"
                        strokeWidth={2}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="scenarios" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Scenario Analysis</CardTitle>
              <CardDescription>
                Compare different timing strategies and their outcomes
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-3">
                <div className="p-4 border rounded-lg">
                  <h4 className="font-medium mb-2">Conservative Strategy</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>Risk Level</span>
                      <Badge className="bg-zinc-100 text-zinc-700">Low</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span>Potential Savings</span>
                      <span>$180,000</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Timeline Impact</span>
                      <span>+1 week</span>
                    </div>
                  </div>
                </div>
                <div className="p-4 border-2 border-primary rounded-lg">
                  <h4 className="font-medium mb-2">Recommended Strategy</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>Risk Level</span>
                      <Badge className="bg-amber-100 text-amber-700">
                        Medium
                      </Badge>
                    </div>
                    <div className="flex justify-between">
                      <span>Potential Savings</span>
                      <span className="text-zinc-600">$360,000</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Timeline Impact</span>
                      <span>+2 weeks</span>
                    </div>
                  </div>
                </div>
                <div className="p-4 border rounded-lg">
                  <h4 className="font-medium mb-2">Aggressive Strategy</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>Risk Level</span>
                      <Badge className="bg-red-100 text-red-700">High</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span>Potential Savings</span>
                      <span>$540,000</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Timeline Impact</span>
                      <span>+4 weeks</span>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
