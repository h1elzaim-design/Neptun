"""Der autonome Hub (ADR-018, Epic #351) — die Box.

Was hier liegt, sind **Regeln als Code**: was ein Agent handeln darf
(`limits`), die Prüfung vor jeder Order (`box`), die Verlustleiter (`ladder`),
die Aktionen des Risk-Agenten (`risk`) und der einzige Weg zum Broker
(`executor`).

Was hier **nicht** liegt: ein Agent, ein Bot, ein Zeitplan. Die Box steht vor
allem anderen — sie ist gebaut, bevor es etwas gibt, das sie einsperren müsste.
"""
