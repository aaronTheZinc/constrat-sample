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
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import {
  TrendingUp,
  TrendingDown,
  Brain,
  Calculator,
  AlertTriangle,
  Target,
  Zap,
} from "lucide-react";

// Mock prediction data
const priceHistoryData = [
  { month: "Jan", steel: 820, lumber: 450, concrete: 120 },
  { month: "Feb", steel: 835, lumber: 425, concrete: 122 },
  { month: "Mar", steel: 847, lumber: 410, concrete: 125 },
  { month: "Apr", steel: 860, lumber: 395, concrete: 128 },
  { month: "May", steel: 875, lumber: 380, concrete: 130 },
  { month: "Jun", steel: 890, lumber: 365, concrete: 132 },
];

const predictionData = [
  { month: "Jul", steel: 905, lumber: 350, concrete: 135, confidence: 85 },
  { month: "Aug", steel: 920, lumber: 340, concrete: 138, confidence: 82 },
  { month: "Sep", steel: 935, lumber: 335, concrete: 140, confidence: 78 },
  { month: "Oct", steel: 950, lumber: 330, concrete: 142, confidence: 75 },
  { month: "Nov", steel: 965, lumber: 325, concrete: 145, confidence: 72 },
  { month: "Dec", steel: 980, lumber: 320, concrete: 148, confidence: 70 },
];

const riskFactors = [
  { name: "Supply Chain", value: 35, color: "#ef4444" },
  { name: "Weather", value: 25, color: "#f59e0b" },
  { name: "Labor Costs", value: 20, color: "#10b981" },
  { name: "Fuel Prices", value: 15, color: "#3b82f6" },
  { name: "Regulations", value: 5, color: "#8b5cf6" },
];

const materialPredictions = [
  {
    material: "Steel Rebar Grade 60",
    currentPrice: 847,
    predictedPrice: 920,
    change: 8.6,
    confidence: 85,
    recommendation: "buy_later",
    optimalTiming: "2 weeks",
    savings: 62,
  },
  {
    material: "Concrete Mix 4000 PSI",
    currentPrice: 125,
    predictedPrice: 138,
    change: 10.4,
    confidence: 82,
    recommendation: "buy_now",
    optimalTiming: "Immediate",
    savings: 0,
  },
  {
    material: "Lumber 2x4x8 SPF",
    currentPrice: 410,
    predictedPrice: 340,
    change: -17.1,
    confidence: 78,
    recommendation: "wait",
    optimalTiming: "3 months",
    savings: 70,
  },
];

export function CostPredictionEngine() {
  const [selectedMaterial, setSelectedMaterial] = useState("steel");
  const [predictionHorizon, setPredictionHorizon] = useState("6months");

  const getRecommendationColor = (recommendation: string) => {
    switch (recommendation) {
      case "buy_now":
        return "bg-red-100 text-red-700";
      case "buy_later":
        return "bg-zinc-100 text-zinc-700";
      case "wait":
        return "bg-blue-100 text-blue-700";
      default:
        return "bg-gray-100 text-gray-700";
    }
  };

  const getRecommendationText = (recommendation: string) => {
    switch (recommendation) {
      case "buy_now":
        return "Buy Now";
      case "buy_later":
        return "Buy Later";
      case "wait":
        return "Wait";
      default:
        return "Monitor";
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-foreground">
            Cost Prediction Engine
          </h1>
          <p className="text-muted-foreground">
            AI-powered cost predictions and market analysis for construction
            materials
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" className="gap-2 bg-transparent">
            <Brain className="h-4 w-4" />
            Retrain Model
          </Button>
          <Button className="gap-2">
            <Calculator className="h-4 w-4" />
            Run Prediction
          </Button>
        </div>
      </div>

      {/* Prediction Controls */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Target className="h-5 w-5" />
            Prediction Parameters
          </CardTitle>
          <CardDescription>
            Configure your cost prediction analysis
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-4">
            <div className="space-y-2">
              <Label htmlFor="material">Material Category</Label>
              <Select
                value={selectedMaterial}
                onValueChange={setSelectedMaterial}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select material" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="steel">Steel & Rebar</SelectItem>
                  <SelectItem value="concrete">Concrete & Cement</SelectItem>
                  <SelectItem value="lumber">Lumber & Wood</SelectItem>
                  <SelectItem value="all">All Materials</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="horizon">Prediction Horizon</Label>
              <Select
                value={predictionHorizon}
                onValueChange={setPredictionHorizon}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select timeframe" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1month">1 Month</SelectItem>
                  <SelectItem value="3months">3 Months</SelectItem>
                  <SelectItem value="6months">6 Months</SelectItem>
                  <SelectItem value="1year">1 Year</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="project-size">Project Size</Label>
              <Input placeholder="Enter project value" />
            </div>
            <div className="space-y-2">
              <Label htmlFor="location">Location</Label>
              <Input placeholder="Enter project location" />
            </div>
          </div>
        </CardContent>
      </Card>

      <Tabs defaultValue="predictions" className="space-y-6">
        <TabsList>
          <TabsTrigger value="predictions">Price Predictions</TabsTrigger>
          <TabsTrigger value="analysis">Risk Analysis</TabsTrigger>
          <TabsTrigger value="recommendations">Recommendations</TabsTrigger>
          <TabsTrigger value="model">Model Performance</TabsTrigger>
        </TabsList>

        <TabsContent value="predictions" className="space-y-6">
          {/* Price Trend Chart */}
          <Card>
            <CardHeader>
              <CardTitle>Price Trend Forecast</CardTitle>
              <CardDescription>
                Historical data and AI predictions for construction materials
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={[...priceHistoryData, ...predictionData]}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="month" />
                    <YAxis />
                    <Tooltip />
                    <Line
                      type="monotone"
                      dataKey="steel"
                      stroke="#ef4444"
                      strokeWidth={2}
                      strokeDasharray="0 0 5 5"
                      name="Steel ($/ton)"
                    />
                    <Line
                      type="monotone"
                      dataKey="lumber"
                      stroke="#10b981"
                      strokeWidth={2}
                      strokeDasharray="0 0 5 5"
                      name="Lumber ($/MBF)"
                    />
                    <Line
                      type="monotone"
                      dataKey="concrete"
                      stroke="#3b82f6"
                      strokeWidth={2}
                      strokeDasharray="0 0 5 5"
                      name="Concrete ($/yard³)"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-4 flex items-center gap-4 text-sm text-muted-foreground">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-0.5 bg-gray-400"></div>
                  <span>Historical Data</span>
                </div>
                <div className="flex items-center gap-2">
                  <div
                    className="w-3 h-0.5 bg-gray-400"
                    style={{ borderTop: "2px dashed" }}
                  ></div>
                  <span>AI Predictions</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Material Predictions Table */}
          <Card>
            <CardHeader>
              <CardTitle>Material Price Predictions</CardTitle>
              <CardDescription>
                Detailed predictions for key construction materials
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {materialPredictions.map((material, index) => (
                  <div
                    key={index}
                    className="flex items-center justify-between p-4 border rounded-lg"
                  >
                    <div className="flex-1">
                      <h4 className="font-medium">{material.material}</h4>
                      <div className="flex items-center gap-4 mt-2 text-sm text-muted-foreground">
                        <span>Current: ${material.currentPrice}</span>
                        <span>Predicted: ${material.predictedPrice}</span>
                        <div className="flex items-center gap-1">
                          {material.change > 0 ? (
                            <TrendingUp className="h-3 w-3 text-red-500" />
                          ) : (
                            <TrendingDown className="h-3 w-3 text-zinc-500" />
                          )}
                          <span
                            className={
                              material.change > 0
                                ? "text-red-500"
                                : "text-zinc-500"
                            }
                          >
                            {material.change > 0 ? "+" : ""}
                            {material.change}%
                          </span>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-4">
                      <div className="text-right">
                        <div className="text-sm text-muted-foreground">
                          Confidence
                        </div>
                        <div className="font-medium">
                          {material.confidence}%
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-sm text-muted-foreground">
                          Optimal Timing
                        </div>
                        <div className="font-medium">
                          {material.optimalTiming}
                        </div>
                      </div>
                      <Badge
                        className={getRecommendationColor(
                          material.recommendation
                        )}
                      >
                        {getRecommendationText(material.recommendation)}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analysis" className="space-y-6">
          <div className="grid gap-6 md:grid-cols-2">
            {/* Risk Factors */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <AlertTriangle className="h-5 w-5" />
                  Risk Factors
                </CardTitle>
                <CardDescription>
                  Factors affecting price volatility
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={riskFactors}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={100}
                        paddingAngle={5}
                        dataKey="value"
                      >
                        {riskFactors.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div className="space-y-2 mt-4">
                  {riskFactors.map((factor, index) => (
                    <div
                      key={index}
                      className="flex items-center justify-between"
                    >
                      <div className="flex items-center gap-2">
                        <div
                          className="w-3 h-3 rounded-full"
                          style={{ backgroundColor: factor.color }}
                        ></div>
                        <span className="text-sm">{factor.name}</span>
                      </div>
                      <span className="text-sm font-medium">
                        {factor.value}%
                      </span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Confidence Metrics */}
            <Card>
              <CardHeader>
                <CardTitle>Prediction Confidence</CardTitle>
                <CardDescription>
                  Model accuracy and reliability metrics
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span>Steel Predictions</span>
                    <span>85%</span>
                  </div>
                  <Progress value={85} className="h-2" />
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span>Lumber Predictions</span>
                    <span>78%</span>
                  </div>
                  <Progress value={78} className="h-2" />
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span>Concrete Predictions</span>
                    <span>82%</span>
                  </div>
                  <Progress value={82} className="h-2" />
                </div>
                <div className="pt-4 border-t">
                  <div className="flex justify-between items-center">
                    <span className="font-medium">Overall Model Accuracy</span>
                    <span className="text-lg font-bold text-zinc-600">
                      81.7%
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Market Volatility */}
          <Card>
            <CardHeader>
              <CardTitle>Market Volatility Analysis</CardTitle>
              <CardDescription>
                Price volatility trends over time
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={priceHistoryData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="month" />
                    <YAxis />
                    <Tooltip />
                    <Area
                      type="monotone"
                      dataKey="steel"
                      stackId="1"
                      stroke="#ef4444"
                      fill="#ef4444"
                      fillOpacity={0.3}
                    />
                    <Area
                      type="monotone"
                      dataKey="lumber"
                      stackId="2"
                      stroke="#10b981"
                      fill="#10b981"
                      fillOpacity={0.3}
                    />
                    <Area
                      type="monotone"
                      dataKey="concrete"
                      stackId="3"
                      stroke="#3b82f6"
                      fill="#3b82f6"
                      fillOpacity={0.3}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="recommendations" className="space-y-6">
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Zap className="h-5 w-5" />
                  Immediate Actions
                </CardTitle>
                <CardDescription>
                  Urgent recommendations for your projects
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <AlertTriangle className="h-4 w-4 text-red-600" />
                    <span className="font-medium text-red-900">
                      Urgent: Concrete Prices Rising
                    </span>
                  </div>
                  <p className="text-sm text-red-700">
                    Purchase concrete materials within 48 hours to avoid 10%
                    price increase
                  </p>
                </div>
                <div className="p-3 bg-zinc-50 border border-zinc-200 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <TrendingDown className="h-4 w-4 text-zinc-600" />
                    <span className="font-medium text-zinc-900">
                      Opportunity: Lumber Decline
                    </span>
                  </div>
                  <p className="text-sm text-zinc-700">
                    Delay lumber purchases by 3 months to save up to 17% on
                    costs
                  </p>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Optimization Summary</CardTitle>
                <CardDescription>
                  Potential savings across all materials
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="text-center">
                  <div className="text-3xl font-bold text-zinc-600">
                    $184,200
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Total Potential Savings
                  </div>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>Steel Optimization</span>
                    <span className="text-zinc-600">$62,000</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Lumber Optimization</span>
                    <span className="text-zinc-600">$98,500</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Concrete Optimization</span>
                    <span className="text-zinc-600">$23,700</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="model" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Model Performance Metrics</CardTitle>
              <CardDescription>
                AI model accuracy and training statistics
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-6 md:grid-cols-3">
                <div className="text-center">
                  <div className="text-2xl font-bold">94.2%</div>
                  <div className="text-sm text-muted-foreground">
                    Training Accuracy
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold">81.7%</div>
                  <div className="text-sm text-muted-foreground">
                    Validation Accuracy
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold">2.3%</div>
                  <div className="text-sm text-muted-foreground">
                    Mean Absolute Error
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
