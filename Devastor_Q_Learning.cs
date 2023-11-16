using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;

namespace DevastorTradeSystem
{
    public class Devastor_QLearning
    {
        public List<Devastor_QState> States { get; private set; }
        public Dictionary<string, Devastor_QState> StateLookup { get; private set; }

        public double Alpha { get; internal set; }
        public double Gamma { get; internal set; }

        public HashSet<string> EndStates { get; private set; }
        public int MaxExploreStepsWithinOneEpisode { get; internal set; } //avoid infinite loop
        public bool ShowWarning { get; internal set; } // show runtime warnings regarding Devastor_Q-learning
        public int Episodes { get; internal set; }
        /// <summary>
        /// Devastor_QLearning [1 -- *] State [1 -- *] Action [1 -- *] ActionResult
        /// </summary>
        public Devastor_QLearning()
        {
            States = new List<Devastor_QState>();
            StateLookup = new Dictionary<string, Devastor_QState>();
            EndStates = new HashSet<string>();

            // Default when not set
            MaxExploreStepsWithinOneEpisode = 1000;
            Episodes = 1000;
            Alpha = 0.1;
            Gamma = 0.9;
            ShowWarning = true;
        }

        public void AddState(Devastor_QState state)
        {
            States.Add(state);
        }

        public void RunTraining()
        {
            Devastor_QMethod.Validate(this);

            /*       
            For each episode: Select random initial state 
            Do while not reach goal state
                Select one among all possible actions for the current state 
                Using this possible action, consider to go to the next state 
                Get maximum Devastor_Q value of this next state based on all possible actions                
                Set the next state as the current state
            */

            // For each episode
            var rand = new Random();
            long maxloopEventCount = 0;

            // Train episodes
            for (long i = 0; i < Episodes; i++)
            {
                long maxloop = 0;
                // Select random initial state          
                int stateIndex = rand.Next(States.Count);
                Devastor_QState state = States[stateIndex];
                Devastor_QAction action = null;
                do
                {
                    if (++maxloop > MaxExploreStepsWithinOneEpisode)
                    {
                        if (ShowWarning)
                        {
                            string msg = string.Format(
                            "{0} !! MAXLOOP state: {1} action: {2}, {3} endstate is to difficult to reach?",
                            ++maxloopEventCount, state, action, "maybe your path setup is wrong or the ");
                            Devastor_QMethod.Log(msg);
                        }

                        break;
                    }

                    // no actions, skip this state
                    if (state.Actions.Count == 0)
                        break;

                    // Selection strategy is random based on probability
                    int index = rand.Next(state.Actions.Count);
                    action = state.Actions[index];

                    // Using this possible action, consider to go to the next state
                    // Pick random Action outcome
                    Devastor_QActionResult nextStateResult = action.PickActionByProbability();
                    string nextStateName = nextStateResult.StateName;

                    double Devastor_Q = nextStateResult.Devastor_QEstimated;
                    double r = nextStateResult.Reward;
                    double maxDevastor_Q = MaxDevastor_Q(nextStateName);

                    // Devastor_Q(s,a)= Devastor_Q(s,a) + alpha * (R(s,a) + gamma * Max(next state, all actions) - Devastor_Q(s,a))
                    double value = Devastor_Q + Alpha * (r + Gamma * maxDevastor_Q - Devastor_Q); // Devastor_Q-learning                  
                    nextStateResult.Devastor_QValue = value; // update

                    // is end state go to next episode
                    if (EndStates.Contains(nextStateResult.StateName))
                        break;

                    // Set the next state as the current state                    
                    state = StateLookup[nextStateResult.StateName];

                } while (true);
            }
        }


        double MaxDevastor_Q(string stateName)
        {
            const double defaultValue = 0;

            if (!StateLookup.ContainsKey(stateName))
                return defaultValue;

            Devastor_QState state = StateLookup[stateName];
            var actionsFromState = state.Actions;
            double? maxValue = null;
            foreach (var nextState in actionsFromState)
            {
                foreach (var actionResult in nextState.ActionsResult)
                {
                    double value = actionResult.Devastor_QEstimated;
                    if (value > maxValue || !maxValue.HasValue)
                        maxValue = value;
                }
            }

            // no update
            if (!maxValue.HasValue && ShowWarning)
                Devastor_QMethod.Log(string.Format("Warning: No MaxDevastor_Q value for stateName {0}",
                    stateName));

            return maxValue.HasValue ? maxValue.Value : defaultValue;
        }

        public void PrintDevastor_QLearningStructure()
        {
            Console.WriteLine("** Devastor_Q-Learning structure **");
            foreach (Devastor_QState state in States)
            {
                Console.WriteLine("State {0}", state.StateName);
                foreach (Devastor_QAction action in state.Actions)
                {
                    Console.WriteLine("  Action " + action.ActionName);
                    Console.Write(action.GetActionResults());
                }
            }
            Console.WriteLine();
        }

        public void ShowPolicy()
        {
            Console.WriteLine("** Show Policy **");
            foreach (Devastor_QState state in States)
            {
                double max = Double.MinValue;
                string actionName = "nothing";
                foreach (Devastor_QAction action in state.Actions)
                {
                    foreach (Devastor_QActionResult actionResult in action.ActionsResult)
                    {
                        if (actionResult.Devastor_QEstimated > max)
                        {
                            max = actionResult.Devastor_QEstimated;
                            actionName = action.ActionName.ToString();
                        }
                    }
                }

                Console.WriteLine(string.Format("From state {0} do action {1}, max Devastor_QEstimated is {2}",
                    state.StateName, actionName, max.Pretty()));
            }
        }
    }

    public class Devastor_QState
    {
        public string StateName { get; private set; }
        public List<Devastor_QAction> Actions { get; private set; }

        public void AddAction(Devastor_QAction action)
        {
            Actions.Add(action);
        }

        public Devastor_QState(string stateName, Devastor_QLearning Devastor_Q)
        {
            Devastor_Q.StateLookup.Add(stateName, this);
            StateName = stateName;
            Actions = new List<Devastor_QAction>();
        }

        public override string ToString()
        {
            return string.Format("StateName {0}", StateName);
        }
    }

    public class Devastor_QAction
    {
        private static readonly Random Rand = new Random();
        public Devastor_QActionName ActionName { get; internal set; }
        public string CurrentState { get; private set; }
        public List<Devastor_QActionResult> ActionsResult { get; private set; }

        public void AddActionResult(Devastor_QActionResult actionResult)
        {
            ActionsResult.Add(actionResult);
        }

        public string GetActionResults()
        {
            var sb = new StringBuilder();
            foreach (Devastor_QActionResult actionResult in ActionsResult)
                sb.AppendLine("     ActionResult " + actionResult);

            return sb.ToString();
        }

        public Devastor_QAction(string currentState, Devastor_QActionName actionName = null)
        {
            CurrentState = currentState;
            ActionsResult = new List<Devastor_QActionResult>();
            ActionName = actionName;
        }

        // The sum of action outcomes must be close to 1
        public void ValidateActionsResultProbability()
        {
            const double epsilon = 0.1;

            if (ActionsResult.Count == 0)
                throw new ApplicationException(string.Format(
                    "ValidateActionsResultProbability is invalid, no action results:\n {0}",
                    this));

            double sum = ActionsResult.Sum(a => a.Probability);
            if (Math.Abs(1 - sum) > epsilon)
                throw new ApplicationException(string.Format(
                    "ValidateActionsResultProbability is invalid:\n {0}", this));
        }

        public Devastor_QActionResult PickActionByProbability()
        {
            double d = Rand.NextDouble();
            double sum = 0;
            foreach (Devastor_QActionResult actionResult in ActionsResult)
            {
                sum += actionResult.Probability;
                if (d <= sum)
                    return actionResult;
            }

            // we might get here if sum probability is below 1.0 e.g. 0.99 
            // and the d random value is 0.999
            if (ActionsResult.Count > 0)
                return ActionsResult.Last();

            throw new ApplicationException(string.Format("No PickAction result: {0}", this));
        }

        public override string ToString()
        {
            double sum = ActionsResult.Sum(a => a.Probability);
            return string.Format("ActionName {0} probability sum: {1} actionResultCount {2}",
                ActionName, sum, ActionsResult.Count);
        }
    }

    public class Devastor_QActionResult
    {
        public string StateName { get; internal set; }
        public string PrevStateName { get; internal set; }
        public double Devastor_QValue { get; internal set; } // Devastor_Q value is stored here        
        public double Probability { get; internal set; }
        public double Reward { get; internal set; }

        public double Devastor_QEstimated
        {
            get { return Devastor_QValue * Probability; }
        }

        public Devastor_QActionResult(Devastor_QAction action, string stateNameNext = null,
            double probability = 1, double reward = 0)
        {
            PrevStateName = action.CurrentState;
            StateName = stateNameNext;
            Probability = probability;
            Reward = reward;
        }

        public override string ToString()
        {
            return string.Format("State {0}, Prob. {1}, Reward {2}, PrevState {3}, Devastor_QE {4}",
                StateName, Probability.Pretty(), Reward, PrevStateName, Devastor_QEstimated.Pretty());
        }
    }

    public class Devastor_QActionName
    {
        public string From { get; private set; }
        public string To { get; private set; }

        public Devastor_QActionName(string from, string to = null)
        {
            From = from;
            To = to;
        }

        public override string ToString()
        {
            return GetActionName();
        }

        public string GetActionName()
        {
            if (To == null)
                return From;
            return Devastor_QMethod.ActionNameFromTo(From, To);
        }
    }

    public static class Devastor_QMethod
    {
        public static void Log(string s)
        {
            Console.WriteLine(s);
        }

        public static readonly CultureInfo CultureEnUs = new CultureInfo("en-US");

        public static string ToStringEnUs(this double d)
        {
            return d.ToString("G", CultureEnUs);
        }

        public static string Pretty(this double d)
        {
            return ToStringEnUs(Math.Round(d, 2));
        }

        public static string ActionNameFromTo(string a, string b)
        {
            return string.Format("from_{0}_to_{1}", a, b);
        }

        public static string EnumToString<T>(this T type)
        {
            return Enum.GetName(typeof(T), type);
        }

        public static void ValidateRange(double d, string origin = null)
        {
            if (d < 0 || d > 1)
            {
                string s = origin ?? string.Empty;
                throw new ApplicationException(string.Format("ValidateRange error: {0} {1}", d, s));
            }
        }

        public static void Validate(Devastor_QLearning Devastor_Q)
        {
            foreach (var state in Devastor_Q.States)
            {
                foreach (var action in state.Actions)
                {
                    action.ValidateActionsResultProbability();
                }
            }
        }
    }
}
