import { SeriesRecords } from "@/server/data/series";
import { NextRequest, NextResponse } from "next/server";

export async function GET(req: NextRequest) {
  return NextResponse.json(SeriesRecords);
}
