import { Entity, PrimaryGeneratedColumn, Column, OneToMany } from "typeorm";
import { PurchaseOrderModel } from "./purchase-order";
import { ProjectTimelineModel } from "./timeline";

@Entity()
export class ProjectModel {
  @PrimaryGeneratedColumn("uuid")
  id: string;

  @Column()
  name: string;

  @Column({ nullable: true })
  clientName: string;

  @Column({ type: "date", nullable: true })
  startDate: Date;

  @Column({ type: "date", nullable: true })
  endDate: Date;

  // Relations
  @OneToMany(() => PurchaseOrderModel, (po) => po.project)
  purchaseOrders: PurchaseOrderModel[];

  @OneToMany(() => ProjectTimelineModel, (timeline) => timeline.project)
  timelineEvents: ProjectModel[];
}
