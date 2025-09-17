import { Entity, PrimaryGeneratedColumn, Column, ManyToOne } from "typeorm";
import { PurchaseOrder } from "./purchase-order";

@Entity()
export class LineItemModel {
  @PrimaryGeneratedColumn("uuid")
  id: string;

  @Column()
  itemName: string; // e.g., "Concrete", "Paint"

  @Column()
  classification: string; // e.g., "Structural", "Finishes"

  @Column()
  classificationCode: string;
  // Example: ST01 (structural steel), FN01 (finishes paint)
  // You’ll use this code to query historical purchase timing/cost

  @Column({ type: "decimal", precision: 12, scale: 2 })
  quantity: number;

  @Column({ type: "decimal", precision: 12, scale: 2 })
  unitPrice: number;

  @Column({ type: "decimal", precision: 12, scale: 2 })
  totalCost: number;

  // Relations
  @ManyToOne(() => PurchaseOrder, (po) => po.lineItems)
  purchaseOrder: PurchaseOrder;
}
