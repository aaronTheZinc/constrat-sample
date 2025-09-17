import { Entity, PrimaryGeneratedColumn, Column, ManyToOne } from "typeorm";
import { ProjectModel } from "./project";

@Entity()
export class ProjectTimelineModel {
  @PrimaryGeneratedColumn("uuid")
  id: string;

  @Column()
  phaseName: string; // e.g., "Framing", "Drywall Install"

  @Column({ type: "date" })
  plannedDate: Date;

  @Column({ type: "date", nullable: true })
  actualDate: Date;

  @Column({ nullable: true })
  notes: string;

  // Relations
  @ManyToOne(() => ProjectModel, (project) => project.timelineEvents)
  project: ProjectModel;
}
