import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import type { RankedRow } from '../pages/ScreenPage';

type Props = {
  ranking: RankedRow[];
};

function LeaderboardChart({ ranking }: Props) {
  return (
    <div className="dravix-card rounded-[1.75rem] p-4">
      <div className="mb-4 text-lg font-light text-[var(--dravix-ink)]">Ranking leaderboard (CFRPS)</div>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={ranking} margin={{ left: 16, right: 16, top: 8, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(118,33,35,0.08)" />
            <XAxis dataKey="rank" tick={{ fill: '#762123', fontSize: 12 }} />
            <YAxis domain={[0, 100]} tick={{ fill: '#762123', fontSize: 12 }} />
            <Tooltip
              formatter={(value: number) => [value.toFixed(1), 'CFRPS risk']}
              labelFormatter={(label) => {
                const row = ranking.find((item) => item.rank === label);
                return row ? `#${row.rank} ${row.name}` : `#${label}`;
              }}
            />
            <Line
              type="monotone"
              dataKey="score"
              stroke="#784F74"
              strokeWidth={3}
              dot={{ r: 4, fill: '#E8967F', stroke: '#784F74', strokeWidth: 2 }}
              activeDot={{ r: 6 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default LeaderboardChart;
