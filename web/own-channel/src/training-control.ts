// A target pause is acknowledged by pressing Continue. Keep that acknowledgement
// for the experiment so subsequent reports above the threshold do not pause it.
export class AccuracyPause {
  reached = false;
  observe(accuracy: number): boolean {
    if (this.reached || !(accuracy >= .99)) return false;
    this.reached = true;
    return true;
  }
}
