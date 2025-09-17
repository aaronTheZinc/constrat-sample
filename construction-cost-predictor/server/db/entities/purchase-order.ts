import {
  Entity,
  PrimaryGeneratedColumn,
  Column,
  ManyToOne,
  OneToMany,
} from "typeorm";
import { ProjectModel } from "./project";
import { LineItemModel } from "./line-item";

@Entity()
export class PurchaseOrderModel {
  @PrimaryGeneratedColumn("uuid")
  id: string;

  @Column()
  quickbooksId: string; // maps to QuickBooks PO Id

  @Column({ type: "date" })
  txnDate: Date;

  @Column()
  vendorId: string; // VendorRef.value

  @Column()
  vendorName: string; // VendorRef.name

  @Column({ type: "decimal", precision: 12, scale: 2 })
  totalAmount: number;

  // Relations
  @ManyToOne(() => ProjectModel, (project) => project.purchaseOrders)
  project: ProjectModel;

  @OneToMany(() => LineItemModel, (item) => item.purchaseOrder)
  lineItems: LineItemModel[];
}
