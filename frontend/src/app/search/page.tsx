"use client";

import { useState } from "react";
import { Search, Clock, Video, Sparkles, ArrowRight } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { VideoStatusBadge } from "@/components/video-status-badge";
import { mockQueryResults } from "@/lib/mock-data";
import type { NLQueryResult } from "@/lib/types";
import { formatTimestamp } from "@/lib/format";
import Link from "next/link";

const exampleQueries = [
  "Show me every time a truck entered the parking lot",
  "When did someone loiter near the emergency exit?",
  "Find all vehicles that arrived after dark",
  "People carrying backpacks in the last 24 hours",
  "Crowd gatherings near the main entrance",
];

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<NLQueryResult[] | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [searchTime, setSearchTime] = useState(0);

  const handleSearch = () => {
    if (!query.trim()) return;
    setIsSearching(true);
    // Simulate API call with mock data
    setTimeout(() => {
      setResults(mockQueryResults);
      setSearchTime(142.5);
      setIsSearching(false);
    }, 800);
  };

  const handleExampleClick = (example: string) => {
    setQuery(example);
    setIsSearching(true);
    setTimeout(() => {
      setResults(mockQueryResults);
      setSearchTime(142.5);
      setIsSearching(false);
    }, 800);
  };

  return (
    <div className="p-6 lg:p-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold tracking-tight">Search</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Natural language search across all processed videos
        </p>
      </div>

      {/* Search Bar */}
      <div className="mx-auto max-w-3xl">
        <div className="relative">
          <Sparkles className="absolute left-4 top-1/2 h-5 w-5 -translate-y-1/2 text-primary" />
          <Input
            placeholder='Try: "Show me when a red truck entered after 9pm"'
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSearch()}
            className="h-14 pl-12 pr-24 text-base bg-card/50 border-border/50 focus:border-primary/50"
          />
          <Button
            onClick={handleSearch}
            disabled={isSearching || !query.trim()}
            className="absolute right-2 top-1/2 -translate-y-1/2 gap-2"
          >
            {isSearching ? (
              <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary-foreground border-t-transparent" />
            ) : (
              <Search className="h-4 w-4" />
            )}
            Search
          </Button>
        </div>

        {/* Example Queries */}
        {!results && (
          <div className="mt-6">
            <p className="mb-3 text-xs font-medium uppercase tracking-wider text-muted-foreground/60">
              Example Queries
            </p>
            <div className="flex flex-wrap gap-2">
              {exampleQueries.map((example) => (
                <button
                  key={example}
                  onClick={() => handleExampleClick(example)}
                  className="rounded-full border border-border/50 bg-card/30 px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:border-primary/30 hover:text-foreground"
                >
                  {example}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Results */}
        {results && (
          <div className="mt-6">
            <div className="mb-4 flex items-center justify-between">
              <p className="text-sm text-muted-foreground">
                <span className="font-mono font-medium text-foreground">
                  {results.length}
                </span>{" "}
                results found
              </p>
              <p className="font-mono text-xs text-muted-foreground/60">
                {searchTime}ms
              </p>
            </div>

            <div className="space-y-3">
              {results.map((result, idx) => (
                <Link
                  key={result.event.event_id}
                  href={`/videos/${result.video_id}`}
                >
                  <Card className="border-border/50 bg-card/50 transition-colors hover:border-primary/30">
                    <CardContent className="p-4">
                      <div className="flex items-start gap-4">
                        {/* Rank */}
                        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-primary/10 font-mono text-sm font-bold text-primary">
                          {idx + 1}
                        </div>

                        {/* Content */}
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-foreground">
                            {result.event.description}
                          </p>
                          <div className="mt-2 flex flex-wrap items-center gap-2">
                            <Badge
                              variant="outline"
                              className="font-mono text-[10px] border-primary/20 text-primary"
                            >
                              {result.event.event_type}
                            </Badge>
                            <Badge
                              variant="outline"
                              className="font-mono text-[10px]"
                            >
                              {result.event.class_name}
                            </Badge>
                            <span className="flex items-center gap-1 text-[10px] text-muted-foreground">
                              <Clock className="h-2.5 w-2.5" />
                              {formatTimestamp(result.event.start_time_sec)}
                              {result.event.end_time_sec &&
                                ` - ${formatTimestamp(result.event.end_time_sec)}`}
                            </span>
                          </div>
                          <div className="mt-2 flex items-center gap-2 text-[10px] text-muted-foreground/60">
                            <Video className="h-2.5 w-2.5" />
                            <span className="font-mono">
                              {result.video_filename}
                            </span>
                            <span>&middot;</span>
                            <span className="font-mono">
                              Relevance: {(result.relevance_score * 100).toFixed(0)}%
                            </span>
                            <span>&middot;</span>
                            <span className="font-mono">
                              Confidence: {(result.event.confidence * 100).toFixed(0)}%
                            </span>
                          </div>
                        </div>

                        {/* Relevance bar */}
                        <div className="flex shrink-0 flex-col items-end gap-1">
                          <div className="h-1.5 w-16 overflow-hidden rounded-full bg-muted">
                            <div
                              className="h-full rounded-full bg-primary"
                              style={{
                                width: `${result.relevance_score * 100}%`,
                              }}
                            />
                          </div>
                          <ArrowRight className="h-4 w-4 text-muted-foreground/40" />
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </Link>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
