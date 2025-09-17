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
import {
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ComposedChart,
  Area,
  AreaChart,
} from "recharts";
import {
  TrendingUp,
  TrendingDown,
  Globe,
  Activity,
  DollarSign,
  Truck,
  Factory,
  Zap,
  Cloud,
  Sun,
  CloudRain,
} from "lucide-react";

// Mock market data
const marketTrends = [
  { date: "Jan 1", steel: 820, lumber: 450, concrete: 120, oil: 75, labor: 28 },
  {
    date: "Jan 8",
    steel: 825,
    lumber: 445,
    concrete: 121,
    oil: 78,
    labor: 28.5,
  },
  {
    date: "Jan 15",
    steel: 835,
    lumber: 440,
    concrete: 122,
    oil: 76,
    labor: 29,
  },
  {
    date: "Jan 22",
    steel: 847,
    lumber: 425,
    concrete: 125,
    oil: 79,
    labor: 29.2,
  },
  {
    date: "Jan 29",
    steel: 852,
    lumber: 410,
    concrete: 127,
    oil: 82,
    labor: 29.8,
  },
  {
    date: "Feb 5",
    steel: 860,
    lumber: 395,
    concrete: 128,
    oil: 85,
    labor: 30.1,
  },
  {
    date: "Feb 12",
    steel: 875,
    lumber: 380,
    concrete: 130,
    oil: 88,
    labor: 30.5,
  },
];

const regionalData = [
  { region: "Northeast", steel: 875, lumber: 380, concrete: 130, change: 5.2 },
  { region: "Southeast", steel: 820, lumber: 420, concrete: 125, change: 3.1 },
  { region: "Midwest", steel: 840, lumber: 360, concrete: 122, change: 4.8 },
  { region: "Southwest", steel: 860, lumber: 440, concrete: 135, change: 6.1 },
  { region: "West", steel: 890, lumber: 350, concrete: 140, change: 7.3 },
];

const supplyChainData = [
  { factor: "Transportation", impact: 85, trend: "up" },
  { factor: "Raw Materials", impact: 92, trend: "up" },
  { factor: "Manufacturing", impact: 78, trend: "down" },
  { factor: "Inventory", impact: 65, trend: "stable" },
  { factor: "Labor Availability", impact: 88, trend: "up" },
];

const weatherImpact = [
  { date: "Today", temp: 72, precipitation: 0, wind: 8, impact: "low" },
  { date: "Tomorrow", temp: 68, precipitation: 20, wind: 12, impact: "medium" },
  { date: "Day 3", temp: 65, precipitation: 80, wind: 18, impact: "high" },
  { date: "Day 4", temp: 70, precipitation: 60, wind: 15, impact: "medium" },
  { date: "Day 5", temp: 75, precipitation: 10, wind: 10, impact: "low" },
  { date: "Day 6", temp: 78, precipitation: 0, wind: 6, impact: "low" },
  { date: "Day 7", temp: 80, precipitation: 0, wind: 8, impact: "low" },
];

const commodityPrices = [
  { commodity: "Crude Oil", price: 88.45, change: 2.3, impact: "High" },
  { commodity: "Natural Gas", price: 3.12, change: -1.8, impact: "Medium" },
  { commodity: "Iron Ore", price: 145.2, change: 4.1, impact: "High" },
  { commodity: "Copper", price: 8.95, change: 1.7, impact: "Medium" },
  { commodity: "Aluminum", price: 2.34, change: -0.5, impact: "Low" },
];

export function MarketDataVisualization() {
  const [selectedRegion, setSelectedRegion] = useState("all");
  const [timeframe, setTimeframe] = useState("1month");

  const getImpactColor = (impact: string) => {
    switch (impact.toLowerCase()) {
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

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case "up":
        return <TrendingUp className="h-3 w-3 text-red-500" />;
      case "down":
        return <TrendingDown className="h-3 w-3 text-zinc-500" />;
      default:
        return <Activity className="h-3 w-3 text-gray-500" />;
    }
  };

  const getWeatherIcon = (impact: string) => {
    switch (impact) {
      case "high":
        return <CloudRain className="h-4 w-4 text-blue-500" />;
      case "medium":
        return <Cloud className="h-4 w-4 text-gray-500" />;
      default:
        return <Sun className="h-4 w-4 text-yellow-500" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-foreground">
            Market Data Visualization
          </h1>
          <p className="text-muted-foreground">
            Real-time market trends, weather impact, and supply chain analysis
          </p>
        </div>
        <div className="flex gap-2">
          <Select value={selectedRegion} onValueChange={setSelectedRegion}>
            <SelectTrigger className="w-32">
              <SelectValue placeholder="Region" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Regions</SelectItem>
              <SelectItem value="northeast">Northeast</SelectItem>
              <SelectItem value="southeast">Southeast</SelectItem>
              <SelectItem value="midwest">Midwest</SelectItem>
              <SelectItem value="southwest">Southwest</SelectItem>
              <SelectItem value="west">West</SelectItem>
            </SelectContent>
          </Select>
          <Select value={timeframe} onValueChange={setTimeframe}>
            <SelectTrigger className="w-32">
              <SelectValue placeholder="Timeframe" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="1week">1 Week</SelectItem>
              <SelectItem value="1month">1 Month</SelectItem>
              <SelectItem value="3months">3 Months</SelectItem>
              <SelectItem value="1year">1 Year</SelectItem>
            </SelectContent>
          </Select>
          <Button className="gap-2">
            <Activity className="h-4 w-4" />
            Refresh Data
          </Button>
        </div>
      </div>

      {/* Key Market Indicators */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Steel Index</CardTitle>
            <Factory className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">$875/ton</div>
            <p className="text-xs text-red-600 flex items-center">
              <TrendingUp className="h-3 w-3 mr-1" />
              +5.2% this week
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Lumber Index</CardTitle>
            <Truck className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">$380/MBF</div>
            <p className="text-xs text-zinc-600 flex items-center">
              <TrendingDown className="h-3 w-3 mr-1" />
              -8.1% this week
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Fuel Costs</CardTitle>
            <Zap className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">$88.45/bbl</div>
            <p className="text-xs text-red-600 flex items-center">
              <TrendingUp className="h-3 w-3 mr-1" />
              +2.3% this week
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Labor Costs</CardTitle>
            <DollarSign className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">$30.50/hr</div>
            <p className="text-xs text-red-600 flex items-center">
              <TrendingUp className="h-3 w-3 mr-1" />
              +1.8% this week
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Market Volatility
            </CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">Medium</div>
            <p className="text-xs text-amber-600">Elevated activity</p>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="trends" className="space-y-6">
        <TabsList>
          <TabsTrigger value="trends">Market Trends</TabsTrigger>
          <TabsTrigger value="regional">Regional Analysis</TabsTrigger>
          <TabsTrigger value="supply-chain">Supply Chain</TabsTrigger>
          <TabsTrigger value="weather">Weather Impact</TabsTrigger>
          <TabsTrigger value="commodities">Commodities</TabsTrigger>
        </TabsList>

        <TabsContent value="trends" className="space-y-6">
          {/* Market Trends Chart */}
          <Card>
            <CardHeader>
              <CardTitle>Material Price Trends</CardTitle>
              <CardDescription>
                Historical price movements for key construction materials
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={marketTrends}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis yAxisId="left" />
                    <YAxis yAxisId="right" orientation="right" />
                    <Tooltip />
                    <Area
                      yAxisId="left"
                      type="monotone"
                      dataKey="steel"
                      fill="#ef4444"
                      fillOpacity={0.1}
                      stroke="#ef4444"
                      strokeWidth={2}
                    />
                    <Line
                      yAxisId="left"
                      type="monotone"
                      dataKey="lumber"
                      stroke="#10b981"
                      strokeWidth={2}
                      dot={false}
                    />
                    <Line
                      yAxisId="right"
                      type="monotone"
                      dataKey="oil"
                      stroke="#f59e0b"
                      strokeWidth={2}
                      dot={false}
                    />
                    <Bar
                      yAxisId="right"
                      dataKey="labor"
                      fill="#3b82f6"
                      fillOpacity={0.3}
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* Market Correlation */}
          <Card>
            <CardHeader>
              <CardTitle>Price Correlation Matrix</CardTitle>
              <CardDescription>
                How different material prices correlate with each other
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-3">
                <div className="p-4 border rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">Steel vs Oil</span>
                    <Badge className="bg-red-100 text-red-700">
                      Strong +0.85
                    </Badge>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Steel prices closely follow oil price movements
                  </div>
                </div>
                <div className="p-4 border rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">
                      Lumber vs Weather
                    </span>
                    <Badge className="bg-amber-100 text-amber-700">
                      Moderate -0.62
                    </Badge>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Lumber prices inversely affected by weather conditions
                  </div>
                </div>
                <div className="p-4 border rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">Labor vs Demand</span>
                    <Badge className="bg-zinc-100 text-zinc-700">
                      Strong +0.78
                    </Badge>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Labor costs increase with construction demand
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="regional" className="space-y-6">
          {/* Regional Price Comparison */}
          <Card>
            <CardHeader>
              <CardTitle>Regional Price Analysis</CardTitle>
              <CardDescription>
                Material costs across different US regions
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={regionalData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="region" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="steel" fill="#ef4444" name="Steel ($/ton)" />
                    <Bar
                      dataKey="lumber"
                      fill="#10b981"
                      name="Lumber ($/MBF)"
                    />
                    <Bar
                      dataKey="concrete"
                      fill="#3b82f6"
                      name="Concrete ($/yard³)"
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* Regional Insights */}
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Highest Cost Regions</CardTitle>
                <CardDescription>
                  Regions with premium material costs
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                {regionalData
                  .sort((a, b) => b.change - a.change)
                  .slice(0, 3)
                  .map((region, index) => (
                    <div
                      key={index}
                      className="flex items-center justify-between p-3 border rounded-lg"
                    >
                      <div>
                        <div className="font-medium">{region.region}</div>
                        <div className="text-sm text-muted-foreground">
                          Steel: ${region.steel} • Lumber: ${region.lumber}
                        </div>
                      </div>
                      <Badge className="bg-red-100 text-red-700">
                        +{region.change}%
                      </Badge>
                    </div>
                  ))}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Cost-Effective Regions</CardTitle>
                <CardDescription>
                  Regions with competitive pricing
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                {regionalData
                  .sort((a, b) => a.change - b.change)
                  .slice(0, 3)
                  .map((region, index) => (
                    <div
                      key={index}
                      className="flex items-center justify-between p-3 border rounded-lg"
                    >
                      <div>
                        <div className="font-medium">{region.region}</div>
                        <div className="text-sm text-muted-foreground">
                          Steel: ${region.steel} • Lumber: ${region.lumber}
                        </div>
                      </div>
                      <Badge className="bg-zinc-100 text-zinc-700">
                        +{region.change}%
                      </Badge>
                    </div>
                  ))}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="supply-chain" className="space-y-6">
          {/* Supply Chain Health */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Globe className="h-5 w-5" />
                Supply Chain Health Index
              </CardTitle>
              <CardDescription>
                Current status of key supply chain factors
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {supplyChainData.map((factor, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-3 border rounded-lg"
                >
                  <div className="flex items-center gap-3">
                    {getTrendIcon(factor.trend)}
                    <div>
                      <div className="font-medium">{factor.factor}</div>
                      <div className="text-sm text-muted-foreground">
                        Impact Level: {factor.impact}%
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-24 bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-primary h-2 rounded-full"
                        style={{ width: `${factor.impact}%` }}
                      ></div>
                    </div>
                    <span className="text-sm font-medium">
                      {factor.impact}%
                    </span>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>

          {/* Supply Chain Alerts */}
          <Card>
            <CardHeader>
              <CardTitle>Supply Chain Alerts</CardTitle>
              <CardDescription>
                Current disruptions and potential issues
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
                <div className="flex items-center gap-2 mb-1">
                  <TrendingUp className="h-4 w-4 text-red-600" />
                  <span className="font-medium text-red-900">
                    Transportation Delays
                  </span>
                </div>
                <p className="text-sm text-red-700">
                  Port congestion causing 3-5 day delays in steel shipments from
                  overseas suppliers
                </p>
              </div>
              <div className="p-3 bg-amber-50 border border-amber-200 rounded-lg">
                <div className="flex items-center gap-2 mb-1">
                  <Activity className="h-4 w-4 text-amber-600" />
                  <span className="font-medium text-amber-900">
                    Labor Shortage
                  </span>
                </div>
                <p className="text-sm text-amber-700">
                  Manufacturing capacity reduced by 15% due to skilled labor
                  shortages in key regions
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="weather" className="space-y-6">
          {/* Weather Forecast */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Cloud className="h-5 w-5" />
                7-Day Weather Impact Forecast
              </CardTitle>
              <CardDescription>
                Weather conditions affecting construction activities
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-3 md:grid-cols-7">
                {weatherImpact.map((day, index) => (
                  <div
                    key={index}
                    className="p-3 border rounded-lg text-center"
                  >
                    <div className="text-sm font-medium mb-2">{day.date}</div>
                    <div className="flex justify-center mb-2">
                      {getWeatherIcon(day.impact)}
                    </div>
                    <div className="text-lg font-bold">{day.temp}°F</div>
                    <div className="text-xs text-muted-foreground">
                      {day.precipitation}% rain
                    </div>
                    <Badge
                      className={`mt-2 ${getImpactColor(day.impact)}`}
                      size="sm"
                    >
                      {day.impact}
                    </Badge>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Weather Impact Analysis */}
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Weather Impact on Materials</CardTitle>
                <CardDescription>
                  How weather affects material availability and pricing
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="p-3 border rounded-lg">
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium">Concrete Operations</span>
                    <Badge className="bg-red-100 text-red-700">High Risk</Badge>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Rain expected Day 3-4 will halt concrete pours, increasing
                    demand surge afterward
                  </p>
                </div>
                <div className="p-3 border rounded-lg">
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium">Steel Delivery</span>
                    <Badge className="bg-amber-100 text-amber-700">
                      Medium Risk
                    </Badge>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    High winds may delay steel beam deliveries and crane
                    operations
                  </p>
                </div>
                <div className="p-3 border rounded-lg">
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium">Lumber Storage</span>
                    <Badge className="bg-zinc-100 text-zinc-700">
                      Low Risk
                    </Badge>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Covered storage protects lumber from weather impact
                  </p>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Seasonal Trends</CardTitle>
                <CardDescription>
                  Historical weather impact patterns
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={weatherImpact}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis />
                      <Tooltip />
                      <Area
                        type="monotone"
                        dataKey="precipitation"
                        stroke="#3b82f6"
                        fill="#3b82f6"
                        fillOpacity={0.3}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="commodities" className="space-y-6">
          {/* Commodity Prices */}
          <Card>
            <CardHeader>
              <CardTitle>Global Commodity Prices</CardTitle>
              <CardDescription>
                Key commodities affecting construction material costs
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {commodityPrices.map((commodity, index) => (
                  <div
                    key={index}
                    className="flex items-center justify-between p-4 border rounded-lg"
                  >
                    <div className="flex items-center gap-3">
                      <DollarSign className="h-5 w-5 text-muted-foreground" />
                      <div>
                        <div className="font-medium">{commodity.commodity}</div>
                        <div className="text-sm text-muted-foreground">
                          Construction Impact: {commodity.impact}
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-bold">
                        ${commodity.price}
                      </div>
                      <div
                        className={`text-sm flex items-center ${
                          commodity.change > 0
                            ? "text-red-600"
                            : "text-zinc-600"
                        }`}
                      >
                        {commodity.change > 0 ? (
                          <TrendingUp className="h-3 w-3 mr-1" />
                        ) : (
                          <TrendingDown className="h-3 w-3 mr-1" />
                        )}
                        {commodity.change > 0 ? "+" : ""}
                        {commodity.change}%
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
