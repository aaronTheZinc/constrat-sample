import { Project } from "@/types/project";
import { ProjectModel } from "./entities/project";
import { PurchaseOrderModel } from "./entities/purchase-order";

import { DataSource } from "typeorm";
import { LineItemModel } from "./entities/line-item";
import { ProjectTimelineModel } from "./entities/timeline";

export const datasource = new DataSource({
  type: "mysql",
  host: "localhost",
  port: 3306,
  username: "test",
  password: "test",
  database: "test",
  entities: [
    ProjectModel,
    LineItemModel,
    PurchaseOrderModel,
    ProjectTimelineModel,
  ],
});

export type CreateProjectInput = Omit<ProjectModel, "id"> 

/**
 * Create project & trigger project optimization if purchase orders are synced
 * @param input 
 */
export const createProject = async (input: CreateProjectInput) => {
  const newProject = new ProjectModel();
  newProject.name = input.name;
  newProject.startDate = input.startDate
  const projectInput = datasource.manager.create(ProjectModel, {
    ...input,
  });

  const project =  await datasource.manager.save(projectInput);

  // r.purchaseOrders[0].
};

export const getProjects = async (): Promise<Project[]> => {
  const projects = await datasource.manager.find(ProjectModel);
  return projects;
};
