export interface QuickBooksVendorRef {
  value: string;
  name: string;
}

export interface QuickBooksItemRef {
  value: string;
  name: string;
}

export interface QuickBooksLineDetail {
  ItemRef: QuickBooksItemRef;
  Qty: number;
  UnitPrice: number;
}

export interface QuickBooksLine {
  Id: string;
  DetailType: string;
  Amount: number;
  ItemBasedExpenseLineDetail?: QuickBooksLineDetail;
}

export interface QuickBooksPurchaseOrder {
  Id: string;
  TxnDate: string;
  VendorRef: QuickBooksVendorRef;
  Line: QuickBooksLine[];
  TotalAmt: number;
}

export interface PurchaseOrder extends QuickBooksPurchaseOrder {
  id: string;
  txnDate: string;
}
