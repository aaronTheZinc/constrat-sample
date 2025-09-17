"use client"

import { useState } from "react"
import { DashboardLayout } from "@/components/dashboard-layout"
import { DashboardOverview } from "@/components/dashboard-overview"
import { PurchaseOrderManager } from "@/components/purchase-order-manager"
import { CostPredictionEngine } from "@/components/cost-prediction-engine"
import { MarketDataVisualization } from "@/components/market-data-visualization"
import { TimingOptimizer } from "@/components/timing-optimizer"
import { WeatherForecast } from "@/components/weather-forecast"

export default function HomePage() {
  const [activeSection, setActiveSection] = useState("Overview")

  const renderContent = () => {
    switch (activeSection) {
      case "Purchase Orders":
        return <PurchaseOrderManager />
      case "Cost Predictions":
        return <CostPredictionEngine />
      case "Market Data":
        return <MarketDataVisualization />
      case "Timing Optimizer":
        return <TimingOptimizer />
      case "Weather Forecast":
        return <WeatherForecast />
      default:
        return <DashboardOverview />
    }
  }

  return (
    <DashboardLayout activeSection={activeSection} setActiveSection={setActiveSection}>
      {renderContent()}
    </DashboardLayout>
  )
}
