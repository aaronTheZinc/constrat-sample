"use client";

import type React from "react";

import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import {
  BarChart3,
  FileText,
  TrendingUp,
  Calendar,
  Cloud,
  Menu,
  Building2,
  DollarSign,
} from "lucide-react";
import { cn } from "@/lib/utils";

const navigation = [
  { name: "Overview", href: "#", icon: BarChart3, current: true },
  {
    name: "Purchase Orders",
    href: "#purchase-orders",
    icon: FileText,
    current: false,
  },
  {
    name: "Cost Predictions",
    href: "#predictions",
    icon: TrendingUp,
    current: false,
  },
  { name: "Market Data", href: "#market", icon: DollarSign, current: false },
  { name: "Timing Optimizer", href: "#timing", icon: Calendar, current: false },
  { name: "Weather Forecast", href: "#weather", icon: Cloud, current: false },
];

interface DashboardLayoutProps {
  children: React.ReactNode;
  activeSection: string;
  setActiveSection: (section: string) => void;
}

export function DashboardLayout({
  children,
  activeSection,
  setActiveSection,
}: DashboardLayoutProps) {
  return (
    <div className="min-h-screen bg-background">
      {/* Desktop Sidebar */}
      <div className="hidden lg:fixed lg:inset-y-0 lg:z-50 lg:flex lg:w-72 lg:flex-col">
        <div className="flex grow flex-col gap-y-5 overflow-y-auto bg-sidebar px-6 pb-4 border-r border-sidebar-border">
          <div className="flex h-16 shrink-0 items-center">
            <span className="ml-2 text-xl font-bold text-sidebar-foreground">
              Constrat
            </span>
          </div>
          <nav className="flex flex-1 flex-col">
            <ul role="list" className="flex flex-1 flex-col gap-y-7">
              <li>
                <ul role="list" className="-mx-2 space-y-1">
                  {navigation.map((item) => (
                    <li key={item.name}>
                      <button
                        onClick={() => setActiveSection(item.name)}
                        className={cn(
                          activeSection === item.name
                            ? "bg-sidebar-accent text-sidebar-accent-foreground"
                            : "text-sidebar-foreground hover:text-sidebar-accent-foreground hover:bg-sidebar-accent/50",
                          "group flex gap-x-3 rounded-md p-2 text-sm leading-6 font-medium w-full text-left transition-colors"
                        )}
                      >
                        <item.icon
                          className={cn(
                            activeSection === item.name
                              ? "text-sidebar-accent-foreground"
                              : "text-sidebar-foreground group-hover:text-sidebar-accent-foreground",
                            "h-6 w-6 shrink-0"
                          )}
                          aria-hidden="true"
                        />
                        {item.name}
                      </button>
                    </li>
                  ))}
                </ul>
              </li>
            </ul>
          </nav>
        </div>
      </div>

      {/* Mobile menu */}
      <Sheet>
        <div className="sticky top-0 z-40 flex items-center gap-x-6 bg-background px-4 py-4 shadow-sm sm:px-6 lg:hidden border-b border-border">
          <SheetTrigger asChild>
            <Button
              variant="outline"
              size="icon"
              className="-m-2.5 bg-transparent"
            >
              <span className="sr-only">Open sidebar</span>
              <Menu className="h-6 w-6" aria-hidden="true" />
            </Button>
          </SheetTrigger>
          <div className="flex-1 text-sm font-semibold leading-6 text-foreground">
            {activeSection}
          </div>
        </div>

        <SheetContent side="left" className="w-72 bg-sidebar">
          <div className="flex h-16 shrink-0 items-center px-6">
            <Building2 className="h-8 w-8 text-sidebar-primary" />
            <span className="ml-2 text-xl font-bold text-sidebar-foreground">
              Constrat.AI
            </span>
          </div>
          <nav className="flex flex-1 flex-col px-6">
            <ul role="list" className="flex flex-1 flex-col gap-y-7">
              <li>
                <ul role="list" className="-mx-2 space-y-1">
                  {navigation.map((item) => (
                    <li key={item.name}>
                      <button
                        onClick={() => setActiveSection(item.name)}
                        className={cn(
                          activeSection === item.name
                            ? "bg-sidebar-accent text-sidebar-accent-foreground"
                            : "text-sidebar-foreground hover:text-sidebar-accent-foreground hover:bg-sidebar-accent/50",
                          "group flex gap-x-3 rounded-md p-2 text-sm leading-6 font-medium w-full text-left transition-colors"
                        )}
                      >
                        <item.icon
                          className={cn(
                            activeSection === item.name
                              ? "text-sidebar-accent-foreground"
                              : "text-sidebar-foreground group-hover:text-sidebar-accent-foreground",
                            "h-6 w-6 shrink-0"
                          )}
                          aria-hidden="true"
                        />
                        {item.name}
                      </button>
                    </li>
                  ))}
                </ul>
              </li>
            </ul>
          </nav>
        </SheetContent>
      </Sheet>

      {/* Main content */}
      <div className="lg:pl-72">
        <main className="py-6">
          <div className="px-4 sm:px-6 lg:px-8">{children}</div>
        </main>
      </div>
    </div>
  );
}
