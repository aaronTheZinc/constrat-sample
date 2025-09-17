"use client";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  LineChart,
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
} from "recharts";
import {
  Cloud,
  Sun,
  CloudRain,
  Wind,
  Thermometer,
  Droplets,
  AlertTriangle,
} from "lucide-react";

// Mock weather data
const weeklyForecast = [
  {
    day: "Mon",
    temp: 72,
    humidity: 45,
    precipitation: 0,
    wind: 8,
    constructionIndex: 95,
  },
  {
    day: "Tue",
    temp: 68,
    humidity: 55,
    precipitation: 20,
    wind: 12,
    constructionIndex: 75,
  },
  {
    day: "Wed",
    temp: 65,
    humidity: 80,
    precipitation: 85,
    wind: 18,
    constructionIndex: 25,
  },
  {
    day: "Thu",
    temp: 70,
    humidity: 70,
    precipitation: 60,
    wind: 15,
    constructionIndex: 45,
  },
  {
    day: "Fri",
    temp: 75,
    humidity: 50,
    precipitation: 10,
    wind: 10,
    constructionIndex: 85,
  },
  {
    day: "Sat",
    temp: 78,
    humidity: 40,
    precipitation: 0,
    wind: 6,
    constructionIndex: 100,
  },
  {
    day: "Sun",
    temp: 80,
    humidity: 35,
    precipitation: 0,
    wind: 8,
    constructionIndex: 100,
  },
];

const monthlyTrends = [
  { month: "Jan", avgTemp: 45, precipitation: 3.2, workableDays: 18 },
  { month: "Feb", avgTemp: 52, precipitation: 2.8, workableDays: 20 },
  { month: "Mar", avgTemp: 62, precipitation: 3.5, workableDays: 22 },
  { month: "Apr", avgTemp: 72, precipitation: 4.1, workableDays: 24 },
  { month: "May", avgTemp: 78, precipitation: 3.8, workableDays: 26 },
  { month: "Jun", avgTemp: 85, precipitation: 2.1, workableDays: 28 },
];

const weatherAlerts = [
  {
    type: "severe",
    title: "Heavy Rain Warning",
    description:
      "Thunderstorms expected Wednesday-Thursday. Halt concrete operations.",
    impact: "High",
    affectedActivities: ["Concrete pours", "Excavation", "Roofing"],
  },
  {
    type: "moderate",
    title: "High Wind Advisory",
    description:
      "Winds 15-20 mph Tuesday-Wednesday. Crane operations may be affected.",
    impact: "Medium",
    affectedActivities: ["Crane operations", "Steel erection"],
  },
];

export function WeatherForecast() {
  const getWeatherIcon = (precipitation: number, temp: number) => {
    if (precipitation > 50)
      return <CloudRain className="h-5 w-5 text-blue-500" />;
    if (precipitation > 20) return <Cloud className="h-5 w-5 text-gray-500" />;
    return <Sun className="h-5 w-5 text-yellow-500" />;
  };

  const getConstructionIndex = (index: number) => {
    if (index >= 80)
      return { color: "bg-zinc-100 text-zinc-700", label: "Excellent" };
    if (index >= 60)
      return { color: "bg-amber-100 text-amber-700", label: "Good" };
    if (index >= 40)
      return { color: "bg-orange-100 text-orange-700", label: "Fair" };
    return { color: "bg-red-100 text-red-700", label: "Poor" };
  };

  const getAlertColor = (type: string) => {
    switch (type) {
      case "severe":
        return "bg-red-50 border-red-200";
      case "moderate":
        return "bg-amber-50 border-amber-200";
      default:
        return "bg-blue-50 border-blue-200";
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-foreground">
          Weather Forecast
        </h1>
        <p className="text-muted-foreground">
          Weather impact analysis and construction activity recommendations
        </p>
      </div>

      {/* Weather Alerts */}
      {weatherAlerts.length > 0 && (
        <div className="space-y-3">
          {weatherAlerts.map((alert, index) => (
            <div
              key={index}
              className={`p-4 border rounded-lg ${getAlertColor(alert.type)}`}
            >
              <div className="flex items-start gap-3">
                <AlertTriangle className="h-5 w-5 text-red-600 mt-0.5" />
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <h4 className="font-medium">{alert.title}</h4>
                    <Badge
                      className={
                        alert.type === "severe"
                          ? "bg-red-100 text-red-700"
                          : "bg-amber-100 text-amber-700"
                      }
                    >
                      {alert.impact} Impact
                    </Badge>
                  </div>
                  <p className="text-sm text-muted-foreground mb-2">
                    {alert.description}
                  </p>
                  <div className="text-sm">
                    <span className="font-medium">Affected Activities: </span>
                    {alert.affectedActivities.join(", ")}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      <Tabs defaultValue="weekly" className="space-y-6">
        <TabsList>
          <TabsTrigger value="weekly">7-Day Forecast</TabsTrigger>
          <TabsTrigger value="monthly">Monthly Trends</TabsTrigger>
          <TabsTrigger value="impact">Construction Impact</TabsTrigger>
          <TabsTrigger value="planning">Project Planning</TabsTrigger>
        </TabsList>

        <TabsContent value="weekly" className="space-y-6">
          {/* Weekly Forecast Cards */}
          <div className="grid gap-4 md:grid-cols-7">
            {weeklyForecast.map((day, index) => (
              <Card key={index}>
                <CardContent className="p-4 text-center">
                  <div className="font-medium mb-2">{day.day}</div>
                  <div className="flex justify-center mb-2">
                    {getWeatherIcon(day.precipitation, day.temp)}
                  </div>
                  <div className="text-lg font-bold mb-1">{day.temp}°F</div>
                  <div className="text-xs text-muted-foreground mb-2">
                    {day.precipitation}% rain
                  </div>
                  <Badge
                    className={
                      getConstructionIndex(day.constructionIndex).color
                    }
                    size="sm"
                  >
                    {getConstructionIndex(day.constructionIndex).label}
                  </Badge>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Weather Details Chart */}
          <Card>
            <CardHeader>
              <CardTitle>Detailed Weather Conditions</CardTitle>
              <CardDescription>
                Temperature, precipitation, and wind patterns
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={weeklyForecast}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="day" />
                    <YAxis yAxisId="left" />
                    <YAxis yAxisId="right" orientation="right" />
                    <Tooltip />
                    <Line
                      yAxisId="left"
                      type="monotone"
                      dataKey="temp"
                      stroke="#ef4444"
                      strokeWidth={2}
                      name="Temperature (°F)"
                    />
                    <Line
                      yAxisId="right"
                      type="monotone"
                      dataKey="precipitation"
                      stroke="#3b82f6"
                      strokeWidth={2}
                      name="Precipitation (%)"
                    />
                    <Line
                      yAxisId="right"
                      type="monotone"
                      dataKey="wind"
                      stroke="#10b981"
                      strokeWidth={2}
                      name="Wind (mph)"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="monthly" className="space-y-6">
          {/* Monthly Trends */}
          <Card>
            <CardHeader>
              <CardTitle>Monthly Weather Patterns</CardTitle>
              <CardDescription>
                Historical weather trends and workable days analysis
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={monthlyTrends}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="month" />
                    <YAxis yAxisId="left" />
                    <YAxis yAxisId="right" orientation="right" />
                    <Tooltip />
                    <Area
                      yAxisId="left"
                      type="monotone"
                      dataKey="avgTemp"
                      stroke="#ef4444"
                      fill="#ef4444"
                      fillOpacity={0.3}
                      name="Avg Temperature (°F)"
                    />
                    <Area
                      yAxisId="left"
                      type="monotone"
                      dataKey="precipitation"
                      stroke="#3b82f6"
                      fill="#3b82f6"
                      fillOpacity={0.3}
                      name="Precipitation (inches)"
                    />
                    <Line
                      yAxisId="right"
                      type="monotone"
                      dataKey="workableDays"
                      stroke="#10b981"
                      strokeWidth={3}
                      name="Workable Days"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* Seasonal Analysis */}
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Best Construction Months</CardTitle>
                <CardDescription>
                  Months with optimal weather conditions
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex items-center gap-3">
                    <Sun className="h-5 w-5 text-yellow-500" />
                    <span>June</span>
                  </div>
                  <Badge className="bg-zinc-100 text-zinc-700">
                    28 workable days
                  </Badge>
                </div>
                <div className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex items-center gap-3">
                    <Sun className="h-5 w-5 text-yellow-500" />
                    <span>May</span>
                  </div>
                  <Badge className="bg-zinc-100 text-zinc-700">
                    26 workable days
                  </Badge>
                </div>
                <div className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex items-center gap-3">
                    <Sun className="h-5 w-5 text-yellow-500" />
                    <span>April</span>
                  </div>
                  <Badge className="bg-zinc-100 text-zinc-700">
                    24 workable days
                  </Badge>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Weather Risk Factors</CardTitle>
                <CardDescription>
                  Seasonal construction challenges
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex items-center gap-3">
                    <CloudRain className="h-5 w-5 text-blue-500" />
                    <span>Spring Rains</span>
                  </div>
                  <Badge className="bg-amber-100 text-amber-700">
                    March-April
                  </Badge>
                </div>
                <div className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex items-center gap-3">
                    <Thermometer className="h-5 w-5 text-red-500" />
                    <span>Extreme Heat</span>
                  </div>
                  <Badge className="bg-red-100 text-red-700">July-August</Badge>
                </div>
                <div className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex items-center gap-3">
                    <Wind className="h-5 w-5 text-gray-500" />
                    <span>High Winds</span>
                  </div>
                  <Badge className="bg-amber-100 text-amber-700">
                    Fall-Winter
                  </Badge>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="impact" className="space-y-6">
          {/* Construction Suitability Index */}
          <Card>
            <CardHeader>
              <CardTitle>Construction Suitability Index</CardTitle>
              <CardDescription>
                Daily construction activity recommendations based on weather
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={weeklyForecast}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="day" />
                    <YAxis />
                    <Tooltip />
                    <Bar
                      dataKey="constructionIndex"
                      fill="#10b981"
                      name="Construction Suitability (%)"
                      radius={[4, 4, 0, 0]}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* Activity Recommendations */}
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Recommended Activities</CardTitle>
                <CardDescription>
                  Weather-appropriate construction tasks
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="p-3 bg-zinc-50 border border-zinc-200 rounded-lg">
                  <div className="font-medium text-zinc-900 mb-1">
                    Monday - Tuesday
                  </div>
                  <div className="text-sm text-zinc-700">
                    Ideal for concrete pours, steel erection, roofing work
                  </div>
                </div>
                <div className="p-3 bg-amber-50 border border-amber-200 rounded-lg">
                  <div className="font-medium text-amber-900 mb-1">
                    Wednesday - Thursday
                  </div>
                  <div className="text-sm text-amber-700">
                    Indoor work only, equipment maintenance, planning
                  </div>
                </div>
                <div className="p-3 bg-zinc-50 border border-zinc-200 rounded-lg">
                  <div className="font-medium text-zinc-900 mb-1">
                    Friday - Weekend
                  </div>
                  <div className="text-sm text-zinc-700">
                    All outdoor activities, catch-up work, material deliveries
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Weather Impact on Materials</CardTitle>
                <CardDescription>
                  How weather affects material handling and storage
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex items-center gap-3">
                    <Droplets className="h-4 w-4 text-blue-500" />
                    <span>Concrete</span>
                  </div>
                  <Badge className="bg-red-100 text-red-700">
                    High Risk Wed-Thu
                  </Badge>
                </div>
                <div className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex items-center gap-3">
                    <Wind className="h-4 w-4 text-gray-500" />
                    <span>Steel Erection</span>
                  </div>
                  <Badge className="bg-amber-100 text-amber-700">
                    Caution Tue-Wed
                  </Badge>
                </div>
                <div className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex items-center gap-3">
                    <Sun className="h-4 w-4 text-yellow-500" />
                    <span>Lumber</span>
                  </div>
                  <Badge className="bg-zinc-100 text-zinc-700">
                    Good All Week
                  </Badge>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="planning" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Weather-Based Project Planning</CardTitle>
              <CardDescription>
                Optimize your project schedule based on weather forecasts
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="p-4 border rounded-lg">
                  <h4 className="font-medium mb-2">This Week's Strategy</h4>
                  <div className="text-sm text-muted-foreground mb-3">
                    Focus on weather-sensitive activities early in the week,
                    plan indoor work for mid-week storms
                  </div>
                  <div className="grid gap-2 md:grid-cols-2">
                    <div className="text-sm">
                      <span className="font-medium">Priority Tasks:</span>{" "}
                      Concrete pours, roofing
                    </div>
                    <div className="text-sm">
                      <span className="font-medium">Backup Plans:</span>{" "}
                      Interior finishing, planning
                    </div>
                  </div>
                </div>
                <div className="p-4 border rounded-lg">
                  <h4 className="font-medium mb-2">Long-term Outlook</h4>
                  <div className="text-sm text-muted-foreground">
                    Spring weather patterns suggest increased rainfall in
                    March-April. Plan major concrete work for late February or
                    early May.
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
