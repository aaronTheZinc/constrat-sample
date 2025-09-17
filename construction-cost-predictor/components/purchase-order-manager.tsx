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
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Upload,
  FileText,
  Search,
  Plus,
  Eye,
  Download,
  Package,
  AlertCircle,
} from "lucide-react";

// Mock data for purchase orders
const purchaseOrders = [
  {
    id: "PO-2024-0156",
    projectName: "Downtown Office Complex",
    supplier: "BuildMart Supply Co.",
    totalValue: 485200,
    itemCount: 45,
    status: "processing",
    createdDate: "2024-01-15",
    expectedDelivery: "2024-02-15",
    items: [
      {
        name: "Steel Rebar Grade 60",
        quantity: 2500,
        unit: "lbs",
        unitPrice: 0.85,
        total: 2125,
      },
      {
        name: "Concrete Mix 4000 PSI",
        quantity: 150,
        unit: "yards³",
        unitPrice: 125,
        total: 18750,
      },
      {
        name: "Lumber 2x4x8 SPF",
        quantity: 200,
        unit: "pieces",
        unitPrice: 8.5,
        total: 1700,
      },
    ],
  },
  {
    id: "PO-2024-0155",
    projectName: "Residential Development",
    supplier: "Metro Construction Supply",
    totalValue: 298750,
    itemCount: 32,
    status: "optimized",
    createdDate: "2024-01-12",
    expectedDelivery: "2024-02-10",
    items: [
      {
        name: "Roofing Shingles",
        quantity: 85,
        unit: "squares",
        unitPrice: 145,
        total: 12325,
      },
      {
        name: "Insulation R-30",
        quantity: 120,
        unit: "rolls",
        unitPrice: 65,
        total: 7800,
      },
    ],
  },
  {
    id: "PO-2024-0154",
    projectName: "Highway Bridge Repair",
    supplier: "Industrial Materials Inc.",
    totalValue: 156900,
    itemCount: 28,
    status: "complete",
    createdDate: "2024-01-08",
    expectedDelivery: "2024-01-25",
    items: [
      {
        name: "Structural Steel Beams",
        quantity: 12,
        unit: "pieces",
        unitPrice: 2500,
        total: 30000,
      },
      {
        name: "High-Strength Bolts",
        quantity: 500,
        unit: "pieces",
        unitPrice: 15,
        total: 7500,
      },
    ],
  },
];

export function PurchaseOrderManager() {
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedPO, setSelectedPO] = useState<
    (typeof purchaseOrders)[0] | null
  >(null);

  const filteredOrders = purchaseOrders.filter(
    (po) =>
      po.projectName.toLowerCase().includes(searchTerm.toLowerCase()) ||
      po.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      po.supplier.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const getStatusColor = (status: string) => {
    switch (status) {
      case "processing":
        return "bg-amber-100 text-amber-700";
      case "optimized":
        return "bg-zinc-100 text-zinc-700";
      case "complete":
        return "bg-blue-100 text-blue-700";
      default:
        return "bg-gray-100 text-gray-700";
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "processing":
        return <AlertCircle className="h-3 w-3" />;
      case "optimized":
        return <Package className="h-3 w-3" />;
      case "complete":
        return <FileText className="h-3 w-3" />;
      default:
        return <FileText className="h-3 w-3" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-foreground">
            Purchase Order Management
          </h1>
          <p className="text-muted-foreground">
            Import, analyze, and optimize your construction purchase orders
          </p>
        </div>
        <div className="flex gap-2">
          <Dialog>
            <DialogTrigger asChild>
              <Button variant="outline" className="gap-2 bg-transparent">
                <Upload className="h-4 w-4" />
                Import from QuickBooks
              </Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-md">
              <DialogHeader>
                <DialogTitle>Import Purchase Order</DialogTitle>
                <DialogDescription>
                  Upload a purchase order file from QuickBooks or other
                  accounting software
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4">
                <div className="border-2 border-dashed border-border rounded-lg p-6 text-center">
                  <Upload className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
                  <p className="text-sm text-muted-foreground">
                    Drag and drop your file here, or click to browse
                  </p>
                  <Button variant="outline" className="mt-2 bg-transparent">
                    Choose File
                  </Button>
                </div>
                <div className="text-xs text-muted-foreground">
                  Supported formats: CSV, Excel, QuickBooks IIF, PDF
                </div>
              </div>
            </DialogContent>
          </Dialog>
          <Dialog>
            <DialogTrigger asChild>
              <Button className="gap-2">
                <Plus className="h-4 w-4" />
                Create Manual PO
              </Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-2xl">
              <DialogHeader>
                <DialogTitle>Create Purchase Order</DialogTitle>
                <DialogDescription>
                  Manually create a new purchase order for cost analysis
                </DialogDescription>
              </DialogHeader>
              <div className="grid gap-4 py-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="project-name">Project Name</Label>
                    <Input id="project-name" placeholder="Enter project name" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="supplier">Supplier</Label>
                    <Input id="supplier" placeholder="Enter supplier name" />
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="po-number">PO Number</Label>
                    <Input id="po-number" placeholder="PO-2024-XXXX" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="delivery-date">Expected Delivery</Label>
                    <Input id="delivery-date" type="date" />
                  </div>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="description">Description</Label>
                  <Textarea
                    id="description"
                    placeholder="Enter project description"
                  />
                </div>
                <div className="flex justify-end gap-2">
                  <Button variant="outline">Cancel</Button>
                  <Button>Create PO</Button>
                </div>
              </div>
            </DialogContent>
          </Dialog>
        </div>
      </div>

      {/* Search and Filters */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-col gap-4 md:flex-row md:items-center">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search purchase orders..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10"
              />
            </div>
            <div className="flex gap-2">
              <Select defaultValue="all">
                <SelectTrigger className="w-32">
                  <SelectValue placeholder="Status" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Status</SelectItem>
                  <SelectItem value="processing">Processing</SelectItem>
                  <SelectItem value="optimized">Optimized</SelectItem>
                  <SelectItem value="complete">Complete</SelectItem>
                </SelectContent>
              </Select>
              <Select defaultValue="all">
                <SelectTrigger className="w-32">
                  <SelectValue placeholder="Date" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Dates</SelectItem>
                  <SelectItem value="week">This Week</SelectItem>
                  <SelectItem value="month">This Month</SelectItem>
                  <SelectItem value="quarter">This Quarter</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Purchase Orders Table */}
      <Card>
        <CardHeader>
          <CardTitle>Purchase Orders</CardTitle>
          <CardDescription>
            Manage and analyze your construction purchase orders
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>PO Number</TableHead>
                <TableHead>Project</TableHead>
                <TableHead>Supplier</TableHead>
                <TableHead>Items</TableHead>
                <TableHead>Total Value</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Delivery Date</TableHead>
                <TableHead>Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredOrders.map((po) => (
                <TableRow key={po.id}>
                  <TableCell className="font-medium">{po.id}</TableCell>
                  <TableCell>{po.projectName}</TableCell>
                  <TableCell>{po.supplier}</TableCell>
                  <TableCell>{po.itemCount} items</TableCell>
                  <TableCell>${po.totalValue.toLocaleString()}</TableCell>
                  <TableCell>
                    <Badge className={getStatusColor(po.status)}>
                      {getStatusIcon(po.status)}
                      <span className="ml-1 capitalize">{po.status}</span>
                    </Badge>
                  </TableCell>
                  <TableCell>{po.expectedDelivery}</TableCell>
                  <TableCell>
                    <div className="flex gap-1">
                      <Dialog>
                        <DialogTrigger asChild>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => setSelectedPO(po)}
                          >
                            <Eye className="h-3 w-3" />
                          </Button>
                        </DialogTrigger>
                        <DialogContent className="sm:max-w-4xl">
                          <DialogHeader>
                            <DialogTitle>
                              Purchase Order Details - {selectedPO?.id}
                            </DialogTitle>
                            <DialogDescription>
                              {selectedPO?.projectName} • {selectedPO?.supplier}
                            </DialogDescription>
                          </DialogHeader>
                          {selectedPO && (
                            <Tabs defaultValue="items" className="w-full">
                              <TabsList>
                                <TabsTrigger value="items">Items</TabsTrigger>
                                <TabsTrigger value="analysis">
                                  Cost Analysis
                                </TabsTrigger>
                                <TabsTrigger value="timeline">
                                  Timeline
                                </TabsTrigger>
                              </TabsList>
                              <TabsContent value="items" className="space-y-4">
                                <Table>
                                  <TableHeader>
                                    <TableRow>
                                      <TableHead>Item</TableHead>
                                      <TableHead>Quantity</TableHead>
                                      <TableHead>Unit Price</TableHead>
                                      <TableHead>Total</TableHead>
                                    </TableRow>
                                  </TableHeader>
                                  <TableBody>
                                    {selectedPO.items.map((item, index) => (
                                      <TableRow key={index}>
                                        <TableCell>{item.name}</TableCell>
                                        <TableCell>
                                          {item.quantity} {item.unit}
                                        </TableCell>
                                        <TableCell>${item.unitPrice}</TableCell>
                                        <TableCell>
                                          ${item.total.toLocaleString()}
                                        </TableCell>
                                      </TableRow>
                                    ))}
                                  </TableBody>
                                </Table>
                              </TabsContent>
                              <TabsContent
                                value="analysis"
                                className="space-y-4"
                              >
                                <div className="grid gap-4 md:grid-cols-2">
                                  <Card>
                                    <CardHeader>
                                      <CardTitle className="text-lg">
                                        Cost Breakdown
                                      </CardTitle>
                                    </CardHeader>
                                    <CardContent>
                                      <div className="space-y-2">
                                        <div className="flex justify-between">
                                          <span>Materials</span>
                                          <span>
                                            $
                                            {(
                                              selectedPO.totalValue * 0.75
                                            ).toLocaleString()}
                                          </span>
                                        </div>
                                        <div className="flex justify-between">
                                          <span>Labor</span>
                                          <span>
                                            $
                                            {(
                                              selectedPO.totalValue * 0.15
                                            ).toLocaleString()}
                                          </span>
                                        </div>
                                        <div className="flex justify-between">
                                          <span>Equipment</span>
                                          <span>
                                            $
                                            {(
                                              selectedPO.totalValue * 0.1
                                            ).toLocaleString()}
                                          </span>
                                        </div>
                                      </div>
                                    </CardContent>
                                  </Card>
                                  <Card>
                                    <CardHeader>
                                      <CardTitle className="text-lg">
                                        Optimization Potential
                                      </CardTitle>
                                    </CardHeader>
                                    <CardContent>
                                      <div className="space-y-2">
                                        <div className="flex justify-between">
                                          <span>Current Total</span>
                                          <span>
                                            $
                                            {selectedPO.totalValue.toLocaleString()}
                                          </span>
                                        </div>
                                        <div className="flex justify-between text-zinc-600">
                                          <span>Potential Savings</span>
                                          <span>
                                            -$
                                            {(
                                              selectedPO.totalValue * 0.08
                                            ).toLocaleString()}
                                          </span>
                                        </div>
                                        <div className="flex justify-between font-medium">
                                          <span>Optimized Total</span>
                                          <span>
                                            $
                                            {(
                                              selectedPO.totalValue * 0.92
                                            ).toLocaleString()}
                                          </span>
                                        </div>
                                      </div>
                                    </CardContent>
                                  </Card>
                                </div>
                              </TabsContent>
                              <TabsContent
                                value="timeline"
                                className="space-y-4"
                              >
                                <div className="space-y-4">
                                  <div className="flex items-center gap-3 p-3 border rounded-lg">
                                    <div className="w-2 h-2 bg-zinc-500 rounded-full"></div>
                                    <div className="flex-1">
                                      <p className="font-medium">
                                        Order Created
                                      </p>
                                      <p className="text-sm text-muted-foreground">
                                        {selectedPO.createdDate}
                                      </p>
                                    </div>
                                  </div>
                                  <div className="flex items-center gap-3 p-3 border rounded-lg">
                                    <div className="w-2 h-2 bg-amber-500 rounded-full"></div>
                                    <div className="flex-1">
                                      <p className="font-medium">
                                        Processing & Analysis
                                      </p>
                                      <p className="text-sm text-muted-foreground">
                                        In Progress
                                      </p>
                                    </div>
                                  </div>
                                  <div className="flex items-center gap-3 p-3 border rounded-lg opacity-50">
                                    <div className="w-2 h-2 bg-gray-300 rounded-full"></div>
                                    <div className="flex-1">
                                      <p className="font-medium">
                                        Expected Delivery
                                      </p>
                                      <p className="text-sm text-muted-foreground">
                                        {selectedPO.expectedDelivery}
                                      </p>
                                    </div>
                                  </div>
                                </div>
                              </TabsContent>
                            </Tabs>
                          )}
                        </DialogContent>
                      </Dialog>
                      <Button variant="outline" size="sm">
                        <Download className="h-3 w-3" />
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
