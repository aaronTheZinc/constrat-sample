import { ProjectModel } from "@/server/db/entities/project";

type TimingReccomendation = {};

interface PredictionsOverview {
  timingRecommendations: TimingReccomendation[];
}

export interface Project extends ProjectModel {}
